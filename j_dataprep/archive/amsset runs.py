# list = [1, 2, 6]
d = "G"
# list = [3, 4, 5]
list=[1, 2, 3, 4, 5, 6]

# schiphol_file =f"{d}:/Geomatics/thesis/_amsterdamset/23aug/aug23_schip_climate_qgis.txt"
    # f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_schip_qgis.txt"
met_file =  f"{d}:/Geomatics/thesis/heattryout/preprocess/climatedata/UMEPclimate_oneday.txt" # f"{d}:/Geomatics/thesis/_amsterdamset/23aug/aug23_schip_climate_qgis.txt"
    # f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_qgis.txt"
# f"{d}:/Geomatics/thesis/_amsterdamset/23aug/aug23_qgis.txt"
# f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_schip_qgis.txt"
for i in list:
    loc = i


    # # new gap
    # INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsm_0.tif"
    # INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    # INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    # INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/svfs"
    # INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/shadowmats.npz"
    # INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover.tif"
    # INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
    # INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
    # UTC = 0
    # #
    # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/solweig_gap_clim"
    # INPUT_MET = schiphol_file
    # #
    # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
    #                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
    #
    # test.processAlgorithm()
    # #
    # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/solweig_gap"
    # INPUT_MET = met_file
    #
    # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
    #                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
    #
    # test.processAlgorithm()

#     # bridging files
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm_over.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallheight_over.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallaspect_over.tif"
#     UTC = 0
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/solweig_over_climate"
#     INPUT_MET = met_file
#
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                             INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
#     # with cProfile.Profile() as profiler:
#     test.processAlgorithm()
    #
    # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/solweig_over_climate"
    # INPUT_MET = schiphol_file
    #
    # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
    #                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
    #
    # test.processAlgorithm()


    # og files
    # INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
    # INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    # INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    # INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_og/svfs"
    # INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_og/shadowmats.npz"
    # INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover.tif"
    # INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallheight.tif"
    # INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallaspect.tif"
    # UTC = 0
    #
    # # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/solweig_og"
    # # INPUT_MET = met_file
    # #
    # # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
    # #                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
    # #
    # # test.processAlgorithm()
    #
    #
    # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/solweig_og_climate"
    # INPUT_MET = schiphol_file
    #
    # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
    #                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
    #
    # test.processAlgorithm()

    # # 3d files
    INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
    INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    MULT_DSMS =  f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
    # INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/svfs"
    INPUT_SVF = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/svf/svfs"
    INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/shadowmats.npz"
    INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover.tif"
    INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
    INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
    UTC = 0
    OUTPUT_DIR = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/solweig_655"
    INPUT_MET = met_file

    #INPUT_ANISO=INPUT_ANISO
    test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,  INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_MULT_DSMS=MULT_DSMS)
    test.processAlgorithm_3d()

    # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/solweig_og_climate"
    # INPUT_MET = schiphol_file

    # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,  INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO, INPUT_MULT_DSMS=MULT_DSMS)
    # test.processAlgorithm_3d()



    #
   # test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,  INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO, INPUT_MULT_DSMS=MULT_DSMS)

        # test.processAlgorithm()

    # Print profiling results
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')  # Sort by cumulative time
    # stats.print_stats(20)  # Display the top 20 results

# stats.dump_stats("profile_results_cupy_ani_debug.prof")

"""
# 3d testing
INPUT_DSM = None
INPUT_CDSM = "D:/Geomatics/thesis/gaptesting_database/smaller/case1_veg.tif"
INPUT_MULT_DSMS = "D:/Geomatics/thesis/gaptesting_database/case2/case2_5layers.tif"
INPUT_SVF = "D:/Geomatics/thesis/codetestsvf/3d_layeredtiff/svfs"
INPUT_ANISO ="D:/Geomatics/thesis/codetestsvf/3d_layeredtiff/shadowmats.npz"
INPUT_LC = "D:/Geomatics/thesis/gaptesting_database/case2/case2_5layers_landcover.tif"
INPUT_HEIGHT = "D:/Geomatics/thesis/gaptesting_database/case2/case2_5layers_height.tif"
INPUT_ASPECT = "D:/Geomatics/thesis/gaptesting_database/case2/case2_5layers_aspect.tif"
UTC = 1
OUTPUT_DIR = "D:/Geomatics/thesis/3D_solweig/case2_5layer"
INPUT_MET = "D:/Geomatics/thesis/heattryout/preprocess/climatedata/UMEPclimate_oneday.txt"



test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET, INPUT_MULT_DSMS=INPUT_MULT_DSMS, INPUT_LC=INPUT_LC, INPUT_ANISO=INPUT_ANISO)
"""
#
# INPUT_DSM = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/final_dsm.tif"
# INPUT_DTM = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/final_dtm.tif"
# INPUT_SVF =  "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/svfs"
# INPUT_ANISO = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/shadowmats.npz"
# INPUT_LC = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/landcover.tif"
# INPUT_HEIGHT = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/wallheight.tif"
# INPUT_ASPECT = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/wallaspect_old.tif"
# INPUT_CDSM = None
# UTC = 0
# OUTPUT_DIR =  "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/original_otherfile"
# # INPUT_MET =  "D:/Geomatics/thesis/heattryout/preprocess/climatedata/UMEPclimate_oneday.txt"
# INPUT_MET =  "D:/Geomatics/thesis/_amsterdamset/23aug/test.txt"
#
# test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
# # # with cProfile.Profile() as profiler:
# test.processAlgorithm()
#
# INPUT_ASPECT = "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/wallaspect.tif"
# OUTPUT_DIR =  "D:/Geomatics/thesis/oldwallvsnewwallmethod/option2/new"
#
# test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                         INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
# test.processAlgorithm()
# with cProfile.Profile() as profiler:
#     test.processAlgorithm_3d()

# Print profiling results
# stats = pstats.Stats(profiler)
# stats.sort_stats('cumulative')  # Sort by cumulative time
# stats.print_stats(20)  # Display the top 20 results

# stats.dump_stats("profile_results_cupy_debug_3D.prof")



# SVF

locations = [1] #[1, 2, 3, 4, 5, 6]
d = "G"
for loc in locations:
    #  og
    # INPUT_DSM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
    # INPUT_CDSM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    # OUTPUT_DIR = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_og"
    # OUTPUT_FILE = "profiling/wcstest"
    # INPUT_DTM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    #
    # with cProfile.Profile() as profiler2:
    #     ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE,
    #                                      INPUT_DTM=INPUT_DTM).processAlgorithm()
    #
    # # gap
    # INPUT_DSM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsm_0.tif"
    # INPUT_CDSM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    # OUTPUT_DIR = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole"
    # OUTPUT_FILE = "profiling/wcstest"
    # INPUT_DTM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    #
    # with cProfile.Profile() as profiler2:
    #     ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE,
    #                                      INPUT_DTM=INPUT_DTM).processAlgorithm()
    #
    # # og over
    # INPUT_DSM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm_over.tif"
    # INPUT_CDSM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    # OUTPUT_DIR = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over"
    # OUTPUT_FILE = "profiling/wcstest"
    # INPUT_DTM = f"E:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    #
    # with cProfile.Profile() as profiler2:
    #     ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE,
    #                                      INPUT_DTM=INPUT_DTM).processAlgorithm()

    #  3d
    # INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
    # INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
    # OUTPUT_DIR = f"{d}:/Geomatics/thesis/_svfcheck/ams/location_{loc}/svf"
    # OUTPUT_FILE = f"profiling/wcstest"
    # INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    # INPUT_DSMS =  f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
    INPUT_DSM = f"{d}:/Geomatics/thesis/gaptesting_database/case2/case2_0.tif"
    INPUT_CDSM = None
    OUTPUT_DIR = f"{d}:/Geomatics/thesis/_svfcheck/ams/generic/svf153"
    OUTPUT_FILE = f"profiling/wcstest"
    # INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
    INPUT_DSMS =  f"{d}:/Geomatics/thesis/gaptesting_database/case2/case2_5layers.tif"

    with cProfile.Profile() as profiler2:
        ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE, INPUT_MULT_DSMS=INPUT_DSMS, INPUT_DTM=None, ANISO=True).processAlgorithm_3d()

#
# with cProfile.Profile() as profiler2:
#     ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE, INPUT_DTM=INPUT_DTM ).processAlgorithm()
    # ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE, INPUT_MULT_DSMS=INPUT_DSMS, INPUT_DTM=INPUT_DTM).processAlgorithm_3d()

    # ProcessingSkyViewFactorAlgorithm(INPUT_DSM, INPUT_CDSM, OUTPUT_DIR, OUTPUT_FILE, dsm2=DSM2, dsm3=DSM3).processAlgorithm()

stats3 = pstats.Stats(profiler2)
stats3.sort_stats('cumulative')
print("\nProfiling with veg cap CDSM:\n")
stats3.print_stats(20)
# stats3.dump_stats("profiling/profile_cupy_layered3d.prof")