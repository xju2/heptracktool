#!/bin/bash

PT_CUT=500
rdo=/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/PhaseIIUpgrade/RDO/ATLAS-P2-RUN4-03-00-00/mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.recon.RDO.e8481_s4149_r14700/RDO.33629020._000047.pool.root.1

Reco_tf.py --CA \
    --inputRDOFile $rdo \
    --outputAODFile AOD.root \
    --steering doRAWtoALL \
    --preInclude InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude \
    --preExec="flags.Tracking.ITkMainPass.minPT=[${PT_CUT}]; flags.Tracking.ITkMainPass.minPTSeed=${PT_CUT}" \
    --maxEvents 2
