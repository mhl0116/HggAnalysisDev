import os
import json
import glob
import uproot
import pandas
import numpy
import awkward
import multiprocessing
import time

import selections.photon_selections as photon_selections
import selections.analysis_selections as analysis_selections
import selections.lepton_selections as lepton_selections
import selections.tau_selections as tau_selections

class LoopHelper():
    """
    Class to perform all looping activities: looping through samples,
    filling histograms, making data/MC plots, yield tables,
    writing a single ntuple with all events, etc
    """

    def __init__(self, **kwargs):
        self.samples = kwargs.get("samples")
        self.selections = kwargs.get("selections")
        self.options = kwargs.get("options")
        self.systematics = kwargs.get("systematics")
        self.years = kwargs.get("years").split(",")

        self.output_tag = kwargs.get("output_tag")
        self.output_dir = kwargs.get("output_dir")
        
        self.batch = kwargs.get("batch")
        self.nCores = kwargs.get("nCores")
        self.debug = kwargs.get("debug")
        self.fast = kwargs.get("fast")

        self.do_plots = kwargs.get("do_plots")
        self.do_tables = kwargs.get("do_tables")
        self.do_ntuple = kwargs.get("do_ntuple")

        self.outputs = []

        if self.debug > 0:
            print("[LoopHelper] Creating LoopHelper instance with options:")
            print("\n".join(["{0}={1!r}".format(a, b) for a, b in kwargs.items()]))

        with open(self.options, "r") as f_in:
            options = json.load(f_in)
            for key, info in options.items():
                setattr(self, key, info)

        self.save_branches.append("process_id")
        self.save_branches.append("weight")

        self.branches_data = [branch for branch in self.branches if "gen" not in branch]
        self.save_branches_data = [branch for branch in self.save_branches if "gen" not in branch]
        

        if self.debug > 0:
            print("[LoopHelper] Opening options file: %s" % self.options)
            print("[LoopHelper] Options loaded as:")
            print("\n".join(["{0}={1!r}".format(a, b) for a, b in options.items()]))

        self.load_samples()

    
    def load_file(self, file, tree_name = "Events", data = False):
        with uproot.open(file) as f:
            tree = f[tree_name]
            if data:
                branches = self.branches_data
            else:
                branches = self.branches
            events = tree.arrays(branches, library = "ak", how = "zip") 
            #events = tree.arrays(branches, entry_start = 0, entry_stop = 10000, library = "ak", how = "zip") 
            # library = "ak" to load arrays as awkward arrays for best performance
            # how = "zip" allows us to access arrays as records, e.g. events.Photon
        return events

    def select_events(self, events):
        # Dipho preselection
        events = photon_selections.diphoton_preselection(events, self.debug)
        events.Photon = events.Photon[photon_selections.select_photons(events, self.debug)]

        if self.selections == "HHggTauTau_InclusivePresel":
            events = analysis_selections.ggTauTau_inclusive_preselection(events, self.debug)
            events.Electron = events.Electron[lepton_selections.select_electrons(events, self.debug)]
            events.Muon = events.Muon[lepton_selections.select_muons(events, self.debug)]
            events.Tau = events.Tau[tau_selections.select_taus(events, self.debug)]
            return events 

    def trim_events(self, events, data):
        events = photon_selections.set_photons(events, self.debug)
        events = lepton_selections.set_electrons(events, self.debug)
        events = lepton_selections.set_muons(events, self.debug)
        events = tau_selections.set_taus(events, self.debug)
        if data:
            branches = self.save_branches_data
        else:
            branches = self.save_branches
        trimmed_events = events[branches]
        return trimmed_events


        # TODO: figure out how to trim object branches
        """
        events = photon_selections.set_photons(events, self.debug)

        if data:
            branches = self.save_branches_data
        else:
            branches = self.save_branches

        trimmed_events = events
        #trimmed_events = events[branches] 
        #a = events[["ggMass", "MET_pt"]]
        #b = events.Photon[["pt", "eta"]]
        #trimmed_events = awkward.zip([a,b])
        return trimmed_events
        """

    def load_samples(self):
        with open(self.samples, "r") as f_in:
            self.samples_dict = json.load(f_in)

        if self.debug > 0:
            print("[LoopHelper] Running over the following samples:")
            print("\n".join(["{0}={1!r}".format(a, b) for a, b in self.samples_dict.items()]))

    def chunks(self, files, fpo):
        for i in range(0, len(files), fpo):
            yield files[i : i + fpo]

    def run(self):
        lumi_map = { "2016" : 35.9, "2017" : 41.5, "2018" : 59 } # FIXME: do in a more configurable way

        self.jobs_manager = []

        for sample, info in self.samples_dict.items():
            if self.debug > 0:
                print("[LoopHelper] Running over sample: %s" % sample)
                print("[LoopHelper] details: ", info)

            files = []
            for year, year_info in info.items():
                if year not in self.years:
                    continue
                for path in year_info["paths"]:
                    files += glob.glob(path + "/*.root")

                if len(files) == 0:
                    continue

                job_info = { 
                    "sample" : sample,
                    "process_id" : info["process_id"],
                    "year" : year,
                    "scale1fb" : 1 if sample == "Data" else year_info["metadata"]["scale1fb"],
                    "lumi" : lumi_map[year]
                }

                file_splits = self.chunks(files, info["fpo"])
                job_id = 0
                for file_split in file_splits:
                    job_id += 1
                    if self.fast:
                        if job_id > 1:
                            if self.debug > 0:
                                print("[LoopHelper] --fast option selected, only looping over 1 file for sample: %s (%s)" % (sample, year))
                            break
                    output = self.output_dir + self.selections + "_" + self.output_tag + "_" + sample + "_" + year + "_" + str(job_id) + ".pkl"
                    self.jobs_manager.append({
                        "info" : job_info,
                        "output" : output, 
                        "files" : file_split
                    })
                    self.outputs.append(output)
                
        if self.batch == "local":
            start = time.time()
            if self.debug > 0:
                print("[LoopHelper] Submitting %d jobs locally on %d cores" % (len(self.jobs_manager), self.nCores))
        
            manager = multiprocessing.Manager()
            running_procs = []
            for job in self.jobs_manager:
                print(job)
                running_procs.append(multiprocessing.Process(target = self.loop_sample, args = (job,)))
                running_procs[-1].start()

                while True:
                    do_break = False
                    for i in range(len(running_procs)):
                        if not running_procs[i].is_alive():
                            running_procs.pop(i)
                            do_break = True
                            break
                        if len(running_procs) < self.nCores: # if we have less than nCores jobs running, break infinite loop and add another
                            do_break = True
                            break
                        else:
                            os.system("sleep 5s")
                    if do_break:
                        break

            while len(running_procs) > 0:
                for i in range(len(running_procs)):
                    try:
                        if not running_procs[i].is_alive():
                            running_procs.pop(i)
                    except:
                        continue

            if self.debug > 0:
                elapsed_time = time.time() - start
                print("[LoopHelper] Total time to run %d jobs on %d cores: %.2f minutes" % (len(self.jobs_manager), self.nCores, elapsed_time/60.))
            

        elif self.batch == "dask":
            return
            #TODO
        elif self.batch == "condor":
            return
            #TODO

        self.merge_outputs()
        return

    def write_to_df(self, events, output_name):
        df = awkward.to_pandas(events)
        df.to_pickle(output_name)
        return

    def loop_sample(self, job): 
        info = job["info"]
        sample = info["sample"]
        files = job["files"]
        output = job["output"]

        if self.debug > 0:
            print("[LoopHelper] Running job with parameters", job)

        if sample == "Data":
            data = True
        else:
            data = False
        
        sel_evts = []
        process_id = info["process_id"]

        for file in files:
            if self.debug > 0:
                print("[LoopHelper] Loading file %s" % file)

            events = self.load_file(file, data = data)
            events = self.select_events(events) 
            
            events["process_id"] = numpy.ones(len(events)) * process_id
            if data:
                events["weight"] = numpy.ones(len(events))
            else:
                events["weight"] = events.genWeight * info["scale1fb"] * info["lumi"]

            events = self.trim_events(events, data)
            sel_evts.append(events)

        events_full = awkward.concatenate(sel_evts)
        self.write_to_df(events_full, output)
        return

    def merge_outputs(self):
        master_file = self.output_dir + self.selections + "_" + self.output_tag +  ".pkl"
        master_df = pandas.DataFrame()
        for file in self.outputs:
            df = pandas.read_pickle(file)
            master_df = pandas.concat([master_df, df], ignore_index=True)

        master_df.to_pickle(master_file)
            
