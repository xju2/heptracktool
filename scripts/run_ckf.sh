#!/bin/bash

# shifter --image=beojan/mpicuda9-2:latest --module=cvmfs,gpu ../scripts/run_ckf.sh

PT_CUTS=(200 400 500 700 900 1000)
rdo=/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/PhaseIIUpgrade/RDO/ATLAS-P2-RUN4-03-00-00/mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.recon.RDO.e8481_s4149_r14700/RDO.33629020._000047.pool.root.1
MAX_EVENTS=10

echo "Input RDO file: ${rdo}"

source /global/cfs/cdirs/atlas/scripts/setupATLAS.sh
setupATLAS
export PATH=/cvmfs/sft.cern.ch/lcg/contrib/ninja/1.11.1/Linux-x86_64/bin:$PATH

asetup Athena,main,latest,here

export ATHENA_CORE_NUMBER=8
export ATHENA_PROC_NUMBER=8

for PT_CUT in "${PT_CUTS[@]}"; do
    run_dir="ckf_pt_${PT_CUT}"
    mkdir -p "$run_dir"

    echo "Running CKF with PT cut = ${PT_CUT} MeV in ${run_dir}"

    (
        cd "$run_dir" || exit 1
        cmd=(
            Reco_tf.py
            --CA "all:True"
            --inputRDOFile "$rdo"
            --outputAODFile AOD.root
            --steering doRAWtoALL
            --preInclude InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude
            --preExec="flags.Tracking.ITkMainPass.minPT=[${PT_CUT}]; flags.Tracking.ITkMainPass.minPTSeed=${PT_CUT}"
            --maxEvents $MAX_EVENTS
            --perfmon fullmonmt
        )
        time "${cmd[@]}"
    )
done
