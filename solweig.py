from src.processor.solweig_algorithm_qgisless_cupy import SOLWEIGAlgorithm

# list = [1, 2, 6]
d = "D"

# list = [1, 2, 3, 4, 5, 6]
# fronti =f"{d}:/Geomatics/thesis/_amsterdamset/location"

list = [3, 4, 5]
schiphol_file = f"{d}:/Geomatics/thesis/_amsterdamset/23aug/aug23_schip_climate_qgis.txt" # f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_schip_climatebike_qgis.txt"

for i in list:
    loc = i

    # new gap
    INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsm_0.tif"
    INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/svfs"
    INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/shadowmats.npz"
    INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
    INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
    INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
    UTC = 0
    #
    OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_gap_climate"
    INPUT_MET = schiphol_file
    # #
    test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
                            INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)

    test.processAlgorithm()

    #     # bridging files
    INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm_over.tif"
    INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/svfs"
    INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/shadowmats.npz"
    INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_top.tif"
    INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallheight_over.tif"
    INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallaspect_over.tif"
    UTC = 0
    OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_over_climate"

    test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
                            INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)

    test.processAlgorithm()

    # # 3d files
    INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
    INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    MULT_DSMS = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
    INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/svfs"
    # INPUT_SVF = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/svf/svfs"
    INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/shadowmats.npz"
    INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
    INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
    INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
    UTC = 0
    OUTPUT_DIR = f"{d}:Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_3d_climate"

    # INPUT_ANISO=INPUT_ANISO
    test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
                            INPUT_MET, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_MULT_DSMS=MULT_DSMS,
                            INPUT_ANISO=INPUT_ANISO)
    test.processAlgorithm_3d()
