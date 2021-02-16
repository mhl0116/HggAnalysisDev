import awkward
import numpy
import numba

import selections.selection_utils as utils
from selections import photon_selections

@numba.jit
def select_photon_nb(photon, gHIdx, mgg):
    nEvents = len(photon)
    nPhotons = numpy.int64(0)
    for i in range(nEvents):
        nPhotons += len(photon[i])

    mask_offsets = numpy.empty(nEvents+1, numpy.int64)
    mask_offsets[0] = 0
    mask_contents = numpy.empty(nPhotons, numpy.bool_)
    for i in range(nEvents):
        mask_offsets[i+1] = mask_offsets[i]
        phos = photon[i]
        leadidx = gHIdx[i][0]
        subleadidx = gHIdx[i][1]
        for j in range(len(phos)):
            pho = phos[j] 
            if (pho.mvaID < -0.7) or ((j != leadidx ) and (j != subleadidx)):
                mask_contents[mask_offsets[i+1]] = False
            elif ((j == leadidx) and (pho.pt/mgg[i] < 0.33)):
                mask_contents[mask_offsets[i+1]] = False
            elif ((j == subleadidx) and (pho.pt/mgg[i] < 0.25)):		#changed 0.3 to 0.25 for sub-leading pho
                mask_contents[mask_offsets[i+1]] = False
            else:
                mask_contents[mask_offsets[i+1]] = True 

            mask_offsets[i+1] += 1

    return mask_offsets, mask_contents 

def select_photon(photon, gHIdx, mgg):
    offsets, contents = select_photon_nb(photon, gHIdx, mgg)
    mask_photons_listoffsetarray = awkward.layout.ListOffsetArray64(awkward.layout.Index64(offsets), awkward.layout.NumpyArray(contents) )
    return awkward.Array(mask_photons_listoffsetarray)    

def diphoton_preselection_fromskim(events, gHIdx, photons, options, debug):
    mask_photons = select_photon(photons, gHIdx.gHidx, events.ggMass)
    mask_diphoton = diphoton_preselection_perevent(events, gHIdx.gHidx, photons, options["resonant"])
    mask = mask_diphoton & (awkward.num(photons[mask_photons]) == 2)
    return events[mask], photons[mask_photons][mask]

@numba.jit
def diphoton_preselection_perevent(events, gHIdx, photons, isRes):
    nEvents = len(photons)
    mask_init = numpy.zeros(nEvents, dtype=numpy.int64)
    mask_dipho =  mask_init > 0 # all False
    for i in range(nEvents):
        phoidx1 = gHIdx[i][0]
        phoidx2 = gHIdx[i][1]
        pho1 = photons[i][phoidx1]
        pho2 = photons[i][phoidx2]
        if (pho1.mvaID < -0.7) | (pho2.mvaID < -0.7): continue
        if (pho1.pt/events.ggMass[i] < 0.3) | (pho2.pt/events.ggMass[i] < 0.25): continue
        if ( events.ggMass[i] < 100 ) | ( events.ggMass[i] > 180 ): continue
        if isRes == False :
            if events.ggMass[i] > 120 and events.ggMass[i] < 130 : continue
        mask_dipho[i] = True 

    return mask_dipho 

def diphoton_preselection(events, photons, options, debug):
    # Initialize cut diagnostics tool for debugging
    cut_diagnostics = utils.CutDiagnostics(events = events, debug = debug, cut_set = "[photon_selections.py : diphoton_preselection]")

    selected_photons = photons[photon_selections.select_photons(events, photons, options, debug)]

    ### mgg cut ###
    resonant = options["resonant"]
    if resonant:
        mgg_mask = numpy.array(events.ggMass > options["diphotons"]["mgg_lower"]) & numpy.array(events.ggMass < options["diphotons"]["mgg_upper"])
    else:
        sideband_low = numpy.array(events.ggMass > options["diphotons"]["mgg_lower"]) & numpy.array(events.ggMass < options["diphotons"]["mgg_sideband_lower"])
        sideband_high = numpy.array(events.ggMass > options["diphotons"]["mgg_sideband_upper"]) & numpy.array(events.ggMass < options["diphotons"]["mgg_upper"])
        mgg_mask = sideband_low | sideband_high

    ### pt/mgg cuts ###
    lead_pt_mgg_requirement = (selected_photons.pt / events.ggMass) > options["photons"]["lead_pt_mgg_cut"]
    sublead_pt_mgg_requirement = (selected_photons.pt / events.ggMass) > options["photons"]["sublead_pt_mgg_cut"]

    lead_pt_mgg_cut = awkward.num(selected_photons[lead_pt_mgg_requirement]) >= 1 # at least 1 photon passing lead requirement
    sublead_pt_mgg_cut = awkward.num(selected_photons[sublead_pt_mgg_requirement]) >= 2 # at least 2 photon passing sublead requirement
    pt_mgg_cut = lead_pt_mgg_cut & sublead_pt_mgg_cut

    ### 2 good selected_photons ###
    n_photon_cut = awkward.num(selected_photons) == 2 # can regain a few % of signal if we set to >= 2 (probably e's that are reconstructed as selected_photons)

    all_cuts = mgg_mask & pt_mgg_cut & n_photon_cut
    cut_diagnostics.add_cuts([mgg_mask, pt_mgg_cut, n_photon_cut, all_cuts], ["mgg in [100, 180]" if resonant else "mgg in [100, 120] or [130, 180]", "lead (sublead) pt/mgg > 0.33 (0.25)", "2 good photons", "all"])

    return events[all_cuts], selected_photons[all_cuts]

#TODO: finish full diphoton preselection for sync purposes
def diphoton_preselection_full(events, photons, options, debug):
    cut_diagnostics = utils.CutDiagnostics(events = events, debug = debug, cut_set = "[photon_selections.py : diphoton_preselection]")

    selected_photons = photons[photon_selections.select_photons(events, photons, options, debug)]
