from src.processor.solweig_algorithm_qgisless_cupy import SOLWEIGAlgorithm
from src.preprocessor.skyviewfactor_algorithm_qgisless import ProcessingSkyViewFactorAlgorithm as SVF
import cProfile
import pstats


"optimization tests"

begin = "D:/Geomatics/optimization_tests"
output = "D:/Geomatics/optimization_tests_computer_3d"
D = 'D'
folder_list = ['250', '500', '1000', '1500', '2000', '3000']
for folder in folder_list:
    INPUT_DSM = f"{begin}/{folder}/final_dsm_over.tif"
    INPUT_SVF = f"{begin}/{folder}/svf/svfs"
    INPUT_ANISO = f"{begin}/{folder}/svf/shadowmats.npz"
    INPUT_LC = f"{begin}/{folder}/landcover.tif"
    INPUT_CDSM = None
    INPUT_DSMS = f"{begin}/{folder}/dsms.tif"
    INPUT_HEIGHT = f"{begin}/{folder}/wallheight.tif"
    INPUT_ASPECT = f"{begin}/{folder}/wallaspect.tif"
    UTC = 0
    OUTPUT_DIR_solweig = f"{output}/{folder}/solweig"
    INPUT_MET = f"{D}:/Geomatics/thesis/heattryout/preprocess/climatedata/UMEPclimate_oneday.txt"

    OUTPUT_DIR_SVF = f"{output}/{folder}/svf"

    # SVF
    dump_stats = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/svf_results.prof"

    test_SVF = SVF(INPUT_DSM=INPUT_DSM,INPUT_CDSM=INPUT_CDSM, OUTPUT_DIR=OUTPUT_DIR_SVF, OUTPUT_FILE='output', INPUT_MULT_DSMS=INPUT_DSMS, USE_VEG=False)

    with cProfile.Profile() as profiler:
        test_SVF.processAlgorithm_3d()

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    stats.dump_stats(dump_stats)
    txt_output = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/svf_results.txt"
    with open(txt_output, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

    # SOLWEIG
    dump_stats = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/profile_results.prof"

    test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR_solweig,
                            INPUT_MET, INPUT_LC=INPUT_LC, INPUT_ANISO=INPUT_ANISO, INPUT_MULT_DSMS=INPUT_DSMS)

    with cProfile.Profile() as profiler:
        test.processAlgorithm_3d()

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    stats.dump_stats(dump_stats)
    txt_output = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/profile_results.txt"
    with open(txt_output, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

for folder in folder_list:
    INPUT_DSM = f"{begin}/{folder}/final_dsm_over.tif"
    INPUT_SVF = f"{begin}/{folder}/svf_trees/svfs"
    INPUT_ANISO = f"{begin}/{folder}/svf_trees/shadowmats.npz"
    INPUT_LC = f"{begin}/{folder}/landcover.tif"
    INPUT_CDSM = f"{begin}/{folder}/CHM.tif"
    INPUT_DSMS = f"{begin}/{folder}/dsms.tif"
    INPUT_HEIGHT = f"{begin}/{folder}/wallheight.tif"
    INPUT_ASPECT = f"{begin}/{folder}/wallaspect.tif"
    UTC = 0
    OUTPUT_DIR_solweig = f"{output}/{folder}/solweig_chm"
    INPUT_MET = f"{D}:/Geomatics/thesis/heattryout/preprocess/climatedata/UMEPclimate_oneday.txt"

    OUTPUT_DIR_SVF = f"{output}/{folder}/svf"

    # SVF
    dump_stats = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/svf_results_chm.prof"

    test_SVF = SVF(INPUT_DSM=INPUT_DSM, INPUT_CDSM=INPUT_CDSM, OUTPUT_DIR=OUTPUT_DIR_SVF, OUTPUT_FILE='output',
                   INPUT_MULT_DSMS=INPUT_DSMS, USE_VEG=True)

    with cProfile.Profile() as profiler:
        test_SVF.processAlgorithm_3d()

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    stats.dump_stats(dump_stats)
    txt_output = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/svf_results_chm.txt"
    with open(txt_output, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

    # SOLWEIG

    dump_stats = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/profile_results_chm.prof"

    test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR_solweig,
                            INPUT_MET, INPUT_LC=INPUT_LC, INPUT_ANISO=INPUT_ANISO)

    with cProfile.Profile() as profiler:
        test.processAlgorithm()

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    stats.dump_stats(dump_stats)
    txt_output = f"{D}:/Geomatics/optimization_tests_computer_3d/{folder}/profile_results_chm.txt"
    with open(txt_output, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(20)




# i = 2
# fronti = "D:/Geomatics/thesis/_amsterdamset/3dtest"
# met_data = f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_schip_climatebike_qgis.txt"
# loc = '1'
# d = "D"


# ['historisch', 'tuindorp', 'vinex', 'volkswijk', 'bloemkool']


#
# nbh_type = "chmfix"
# INPUT_DSM = f"{start}//{nbh_type}/dsm.tif"
# OUTPUT_DIR = f"{start}//{nbh_type}/solweig_green_TEST"
# OUTPUT_FILE = f"profiling/wcstest"
# INPUT_DTM = f"{start}//{nbh_type}/dtm.tif"
# INPUT_CDSM = f"{start}//{nbh_type}/chm.tif"
#
# INPUT_SVF = f"{start}//{nbh_type}/svf/svfs"
# INPUT_ANISO = f"{start}//{nbh_type}/svf/shadowmats.npz"
# INPUT_LC = f"{start}//{nbh_type}/landcover.tif"
# INPUT_HEIGHT = f"{start}//{nbh_type}/height.tif"
# INPUT_ASPECT = f"{start}//{nbh_type}/aspect.tif"
# UTC = 0
# INPUT_MET = "../j_dataprep/climate/avgday_30plus_qgis.txt"
#
# # INPUT_ANISO=INPUT_ANISO
# test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                         INPUT_ANISO=INPUT_ANISO, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM)
# test.processAlgorithm()
#

# d = "G"

# j = 0
# input_mets = ["../j_dataprep/climate/extday_30plus_qgis.txt", "../j_dataprep/climate/avgday_30plus_qgis.txt"]
#
# folders_end = ['ext', 'avg']
# start = 'E:/Geomatics/thesis/_analysisfinalfurther'
#
# for met in input_mets:
#     input_met = met
#     end = folders_end[j]
#
#     for nbh_type in ['historisch', 'tuindorp', 'vinex', 'volkswijk', 'bloemkool']:
#         for i in [0, 1, 2, 3, 4, 5]:
#             INPUT_DSM = f"{start}/{nbh_type}/loc_{i}/final_dsm_over.tif"
#             OUTPUT_DIR = f"{start}//{nbh_type}/loc_{i}/solweig_{end}"
#             OUTPUT_FILE = f"profiling/wcstest"
#             INPUT_DTM = f"{start}/{nbh_type}/loc_{i}/final_dtm.tif"
#             INPUT_CDSM = None  # f"E:/Geomatics/thesis/_analysisfinal/{nbh_type}/loc_{i}/CHM.tif"
#
#             INPUT_SVF = f"{start}//{nbh_type}/loc_{i}/svf_build/svfs"
#             INPUT_ANISO = f"{start}//{nbh_type}/loc_{i}/svf_build/shadowmats.npz"
#             INPUT_LC = f"{start}//{nbh_type}/loc_{i}/landcover_stone.tif"
#             INPUT_HEIGHT = f"{start}//{nbh_type}/loc_{i}/height.tif"
#             INPUT_ASPECT = f"{start}//{nbh_type}/loc_{i}/aspect.tif"
#             UTC = 0
#             INPUT_MET = input_met
#
#             # INPUT_ANISO=INPUT_ANISO
#             test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                                     INPUT_MET, INPUT_ANISO=INPUT_ANISO, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM)
#             test.processAlgorithm()
#
#     for nbh_type in ['historisch', 'tuindorp', 'vinex', 'volkswijk', 'bloemkool']:
#         for i in [0, 1, 2, 3, 4, 5]:
#             INPUT_DSM = f"{start}//{nbh_type}/loc_{i}/final_dsm_over.tif"
#             OUTPUT_DIR = f"{start}//{nbh_type}/loc_{i}/solweig_green_{end}"
#             OUTPUT_FILE = f"profiling/wcstest"
#             INPUT_DTM = f"{start}//{nbh_type}/loc_{i}/final_dtm.tif"
#             INPUT_CDSM = f"{start}//{nbh_type}/loc_{i}/CHM.tif"
#
#             INPUT_SVF = f"{start}//{nbh_type}/loc_{i}/svf/svfs"
#             INPUT_ANISO = f"{start}//{nbh_type}/loc_{i}/svf/shadowmats.npz"
#             INPUT_LC = f"{start}//{nbh_type}/loc_{i}/landcover_stone.tif"
#             INPUT_HEIGHT = f"{start}//{nbh_type}/loc_{i}/height.tif"
#             INPUT_ASPECT = f"{start}//{nbh_type}/loc_{i}/aspect.tif"
#             UTC = 0
#             INPUT_MET = input_met
#
#             # INPUT_ANISO=INPUT_ANISO
#             test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                                     INPUT_MET, INPUT_ANISO=INPUT_ANISO, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM)
#             test.processAlgorithm()
#
#     for nbh_type in ['stedelijk']:
#         for i in [0, 1, 2, 3]:
#             INPUT_DSM = f"{start}/{nbh_type}/loc_{i}/final_dsm_over.tif"
#             OUTPUT_DIR = f"{start}//{nbh_type}/loc_{i}/solweig_{end}"
#             OUTPUT_FILE = f"profiling/wcstest"
#             INPUT_DTM = f"{start}/{nbh_type}/loc_{i}/final_dtm.tif"
#             INPUT_CDSM = None  # f"E:/Geomatics/thesis/_analysisfinal/{nbh_type}/loc_{i}/CHM.tif"
#
#             INPUT_SVF = f"{start}//{nbh_type}/loc_{i}/svf_build/svfs"
#             INPUT_ANISO = f"{start}//{nbh_type}/loc_{i}/svf_build/shadowmats.npz"
#             INPUT_LC = f"{start}//{nbh_type}/loc_{i}/landcover_stone.tif"
#             INPUT_HEIGHT = f"{start}//{nbh_type}/loc_{i}/height.tif"
#             INPUT_ASPECT = f"{start}//{nbh_type}/loc_{i}/aspect.tif"
#             UTC = 0
#             INPUT_MET = input_met
#
#             # INPUT_ANISO=INPUT_ANISO
#             test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                                     INPUT_MET, INPUT_ANISO=INPUT_ANISO, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM)
#             test.processAlgorithm()
#
#     for nbh_type in ['stedelijk']:
#         for i in [0, 1, 2, 3]:
#             INPUT_DSM = f"{start}//{nbh_type}/loc_{i}/final_dsm_over.tif"
#             OUTPUT_DIR = f"{start}//{nbh_type}/loc_{i}/solweig_green_{end}"
#             OUTPUT_FILE = f"profiling/wcstest"
#             INPUT_DTM = f"{start}//{nbh_type}/loc_{i}/final_dtm.tif"
#             INPUT_CDSM = f"{start}//{nbh_type}/loc_{i}/CHM.tif"
#
#             INPUT_SVF = f"{start}//{nbh_type}/loc_{i}/svf/svfs"
#             INPUT_ANISO = f"{start}//{nbh_type}/loc_{i}/svf/shadowmats.npz"
#             INPUT_LC = f"{start}//{nbh_type}/loc_{i}/landcover_stone.tif"
#             INPUT_HEIGHT = f"{start}//{nbh_type}/loc_{i}/height.tif"
#             INPUT_ASPECT = f"{start}//{nbh_type}/loc_{i}/aspect.tif"
#             UTC = 0
#             INPUT_MET = input_met
#
#             # INPUT_ANISO=INPUT_ANISO
#             test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                                     INPUT_MET, INPUT_ANISO=INPUT_ANISO, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM)
#             test.processAlgorithm()
#     j += 1





''' AMSTERDAM SET'''
# list = [1, 2, 6]
# d = "D"
# # i = 2
# fronti = "D:/Geomatics/thesis/_amsterdamset/3dtest"
# met_data =f"{fronti}/sep12_schip_climatebike_qgis.txt" #f'{fronti}/UMEPclimate_oneday.txt'
# >>>>>>> Stashed changes
#
# list2 = [3, 4, 5]
# schiphol_file = f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_schip_climatebike_qgis.txt"
#
# for i in list1:
#     loc = i
# =======
# # mult_dsms_path = (f'{fronti}/dsms.tif')
# # dtm_path = f'{fronti}/dtm.tif'
# # dsm_path = f'{fronti}/dsm.tif'
# # chm_path = f'{fronti}/chm.tif'
# # landcover_path = f'{fronti}/landcover.tif'
# # walla_path = f'{fronti}/wallaspect.tif'
# # wallh_path = f'{fronti}/wallheight2.tif'
# # OUTPUT_DIR_SVF = f'{fronti}/2d'
# # OUTPUT_FILE_SVF = f'{fronti}/svf2.tif'
# # loc = 2
# # INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
# # INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
# # INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
# # MULT_DSMS = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
# # INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/svfs"
# # # INPUT_SVF = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/svf/svfs"
# # INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/shadowmats.npz"
# # INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
# # INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
# # INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
# # UTC = 0
# # OUTPUT_DIR = f"{d}:Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_3d_climate"
#
#     # new gap
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsm_0.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
#     UTC = 0
#     #
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new_build0/solweig_gap_climate"
#     INPUT_MET = schiphol_file
#     # #
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                             INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
#
#     test.processAlgorithm()
#
#     #     # bridging files
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm_over.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_top.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallheight_over.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallaspect_over.tif"
#     UTC = 0
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new_build0/solweig_over_climate"
#
# # mult_dsms_path = (f'{fronti}/dsm.tif')
# # SOLWEIGAlgorithm(INPUT_DSM=dsm_path, INPUT_SVF=f'{fronti}/svfs', INPUT_CDSM=chm_path,  INPUT_HEIGHT=wallh_path, INPUT_ASPECT=walla_path,
# #                  UTC=0, OUTPUT_DIR=f'{fronti}/_amstestone2_2d', INPUT_MET=met_data, INPUT_DTM=dtm_path, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS=mult_dsms_path, INPUT_LC=landcover_path,  INPUT_DEM=None, INPUT_ANISO=f'{fronti}/shadowmats.npz',
# #                  CONIFER_TREES=False, INPUT_THEIGHT=25, INPUT_TDSM=None, TRANS_VEG=15, LEAF_START=97, LEAF_END=300,
# #                  USE_LC_BUILD=True, SAVE_BUILD=False, ALBEDO_WALLS=0.2, ALBEDO_GROUND=0.15,
# #                  EMIS_WALLS=0.9, EMIS_GROUND=0.95, ABS_S=0.7, ABS_L=0.95, POSTURE=0,  ONLYGLOBAL=True,
# #                  OUTPUT_TMRT=True, OUTPUT_LUP=True, OUTPUT_KUP=True, OUTPUT_KDOWN=True, OUTPUT_LDOWN=True, OUTPUT_SH=True, OUTPUT_TREEPLANTER=False,
# #                  CYL=True, version='dsm').processAlgorithm()
#
#
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                             INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
#
#     test.processAlgorithm()
#
#     # # 3d files
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     MULT_DSMS = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/svfs"
#     # INPUT_SVF = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/svf/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
#     UTC = 0
#     OUTPUT_DIR = f"{d}:Geomatics/thesis/_amsterdamset/location_{loc}/new_build0/solweig_3d_climate"
#
#     # INPUT_ANISO=INPUT_ANISO
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                             INPUT_MET, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_MULT_DSMS=MULT_DSMS,
#                             INPUT_ANISO=INPUT_ANISO)
#     test.processAlgorithm_3d()
#
# schiphol_file = f"{d}:/Geomatics/thesis/_amsterdamset/23aug/aug23_schip_climate_qgis.txt"
# for i in list2:
#     loc = i
#
#     # new gap
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsm_0.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_hole/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
#     UTC = 0
#     #
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new_build0/solweig_gap_climate"
#     INPUT_MET = schiphol_file
#     # #
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                             INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
#
#     test.processAlgorithm()
#
#     #     # bridging files
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm_over.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/svf_over/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_top.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallheight_over.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/wallaspect_over.tif"
#     UTC = 0
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new_build0/solweig_over_climate"
#
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR, INPUT_MET,
#                             INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_ANISO=INPUT_ANISO)
#
#     test.processAlgorithm()
#
#     # # 3d files
#     INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
#     INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
#     INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
#     MULT_DSMS = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
#     INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/svfs"
#     # INPUT_SVF = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/svf/svfs"
#     INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/shadowmats.npz"
#     INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
#     INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
#     INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
#     UTC = 0
#     OUTPUT_DIR = f"{d}:Geomatics/thesis/_amsterdamset/location_{loc}/new_build0/solweig_3d_climate"
#
#     # INPUT_ANISO=INPUT_ANISO
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                             INPUT_MET, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_MULT_DSMS=MULT_DSMS,
#                             INPUT_ANISO=INPUT_ANISO)
#     test.processAlgorithm_3d()
