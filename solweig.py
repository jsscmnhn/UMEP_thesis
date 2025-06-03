from src.processor.solweig_algorithm_qgisless_cupy import SOLWEIGAlgorithm
from src.preprocessor.skyviewfactor_algorithm_qgisless import ProcessingSkyViewFactorAlgorithm as SVF

# list = [1, 2, 6]
d = "D"
# i = 2
fronti = "D:/Geomatics/thesis/_amsterdamset/3dtest"
met_data =f"{fronti}/sep12_schip_climatebike_qgis.txt" #f'{fronti}/UMEPclimate_oneday.txt'

# list = [1, 2, 3, 4, 5, 6]

mult_dsms_path = (f'{fronti}/dsms.tif')
dtm_path = f'{fronti}/dtm.tif'
dsm_path = f'{fronti}/dsm.tif'
chm_path = f'{fronti}/chm.tif'
landcover_path = f'{fronti}/landcover.tif'
walla_path = f'{fronti}/wallaspect.tif'
wallh_path = f'{fronti}/wallheight2.tif'
OUTPUT_DIR_SVF = f'{fronti}/2d'
OUTPUT_FILE_SVF = f'{fronti}/svf2.tif'
# loc = 2
# INPUT_DSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dsm.tif"
# INPUT_CDSM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/CHM.tif"
# INPUT_DTM = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/final_dtm.tif"
# MULT_DSMS = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/dsms.tif"
# INPUT_SVF = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/svfs"
# # INPUT_SVF = f"{d}:Geomatics/thesis/_svfcheck/ams/location_{loc}/svf/svfs"
# INPUT_ANISO = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/svf/shadowmats.npz"
# INPUT_LC = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/original/landcover_down.tif"
# INPUT_HEIGHT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallheight.tif"
# INPUT_ASPECT = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/3d/wallaspect.tif"
# UTC = 0
# OUTPUT_DIR = f"{d}:Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_3d_climate"

# SVF(INPUT_DSM=dsm_path, INPUT_CDSM=chm_path, OUTPUT_DIR=OUTPUT_DIR_SVF, OUTPUT_FILE=OUTPUT_FILE_SVF, INPUT_DTM=dtm_path, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS = mult_dsms_path, INPUT_TDSM=None, USE_VEG=True, TRANS_VEG=15, TSDM_EXIST=False, INPUT_THEIGHT=25.0, ANISO=True).processAlgorithm()
# SOLWEIGAlgorithm(INPUT_DSM=INPUT_DSM, INPUT_SVF=INPUT_SVF, INPUT_CDSM=INPUT_CDSM,  INPUT_HEIGHT=INPUT_HEIGHT, INPUT_ASPECT=INPUT_ASPECT,
#                  UTC=0, OUTPUT_DIR=f'{fronti}/__amstestmult_full', INPUT_MET=met_data, INPUT_DTM=INPUT_DTM, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS=MULT_DSMS, INPUT_LC=INPUT_LC,  INPUT_DEM=None, INPUT_ANISO=INPUT_ANISO,
#                  CONIFER_TREES=False, INPUT_THEIGHT=25, INPUT_TDSM=None, TRANS_VEG=15, LEAF_START=97, LEAF_END=300,
#                  USE_LC_BUILD=True, SAVE_BUILD=False, ALBEDO_WALLS=0.2, ALBEDO_GROUND=0.15,
#                  EMIS_WALLS=0.9, EMIS_GROUND=0.95, ABS_S=0.7, ABS_L=0.95, POSTURE=0,  ONLYGLOBAL=True,
#                  OUTPUT_TMRT=True, OUTPUT_LUP=True, OUTPUT_KUP=True, OUTPUT_KDOWN=True, OUTPUT_LDOWN=True, OUTPUT_SH=True, OUTPUT_TREEPLANTER=False,
#                  CYL=True, version='dsms_full').processAlgorithm_3d()

# SOLWEIGAlgorithm(INPUT_DSM=dsm_path, INPUT_SVF=f'{fronti}/svfs', INPUT_CDSM=chm_path,  INPUT_HEIGHT=wallh_path, INPUT_ASPECT=walla_path,
#                  UTC=0, OUTPUT_DIR=f'{fronti}/__amstestmult2', INPUT_MET=met_data, INPUT_DTM=dtm_path, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS=f'{fronti}/dsms.tif', INPUT_LC=landcover_path,  INPUT_DEM=None, INPUT_ANISO=f'{fronti}/shadowmats.npz',
#                  CONIFER_TREES=False, INPUT_THEIGHT=25, INPUT_TDSM=None, TRANS_VEG=15, LEAF_START=97, LEAF_END=300,
#                  USE_LC_BUILD=True, SAVE_BUILD=False, ALBEDO_WALLS=0.2, ALBEDO_GROUND=0.15,
#                  EMIS_WALLS=0.9, EMIS_GROUND=0.95, ABS_S=0.7, ABS_L=0.95, POSTURE=0,  ONLYGLOBAL=True,
#                  OUTPUT_TMRT=True, OUTPUT_LUP=True, OUTPUT_KUP=True, OUTPUT_KDOWN=True, OUTPUT_LDOWN=True, OUTPUT_SH=True, OUTPUT_TREEPLANTER=False,
#                  CYL=True, version='dsms').processAlgorithm_3d()

mult_dsms_path = (f'{fronti}/dsm.tif')
SOLWEIGAlgorithm(INPUT_DSM=dsm_path, INPUT_SVF=f'{fronti}/svfs', INPUT_CDSM=chm_path,  INPUT_HEIGHT=wallh_path, INPUT_ASPECT=walla_path,
                 UTC=0, OUTPUT_DIR=f'{fronti}/_amstestone2_2d', INPUT_MET=met_data, INPUT_DTM=dtm_path, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS=mult_dsms_path, INPUT_LC=landcover_path,  INPUT_DEM=None, INPUT_ANISO=f'{fronti}/shadowmats.npz',
                 CONIFER_TREES=False, INPUT_THEIGHT=25, INPUT_TDSM=None, TRANS_VEG=15, LEAF_START=97, LEAF_END=300,
                 USE_LC_BUILD=True, SAVE_BUILD=False, ALBEDO_WALLS=0.2, ALBEDO_GROUND=0.15,
                 EMIS_WALLS=0.9, EMIS_GROUND=0.95, ABS_S=0.7, ABS_L=0.95, POSTURE=0,  ONLYGLOBAL=True,
                 OUTPUT_TMRT=True, OUTPUT_LUP=True, OUTPUT_KUP=True, OUTPUT_KDOWN=True, OUTPUT_LDOWN=True, OUTPUT_SH=True, OUTPUT_TREEPLANTER=False,
                 CYL=True, version='dsm').processAlgorithm()

# SOLWEIGAlgorithm(INPUT_DSM=dsm_path, INPUT_SVF=f'{fronti}/svfs', INPUT_CDSM=chm_path,  INPUT_HEIGHT=wallh_path, INPUT_ASPECT=walla_path,
#                  UTC=0, OUTPUT_DIR=f'{fronti}/ams_solweig2d', INPUT_MET=met_data, INPUT_DTM=dtm_path, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS=mult_dsms_path, INPUT_LC=landcover_path,  INPUT_DEM=None, INPUT_ANISO=f'{fronti}/shadowmats.npz',
#                  CONIFER_TREES=False, INPUT_THEIGHT=25, INPUT_TDSM=None, TRANS_VEG=15, LEAF_START=97, LEAF_END=300,
#                  USE_LC_BUILD=True, SAVE_BUILD=False, ALBEDO_WALLS=0.2, ALBEDO_GROUND=0.15,
#                  EMIS_WALLS=0.9, EMIS_GROUND=0.95, ABS_S=0.7, ABS_L=0.95, POSTURE=0,  ONLYGLOBAL=True,
#                  OUTPUT_TMRT=True, OUTPUT_LUP=True, OUTPUT_KUP=True, OUTPUT_KDOWN=True, OUTPUT_LDOWN=True, OUTPUT_SH=True, OUTPUT_TREEPLANTER=False,
#                  CYL=True).processAlgorithm()

# SOLWEIGAlgorithm(INPUT_DSM=dsm_path, INPUT_SVF=f'{fronti}/2d/svfs', INPUT_CDSM=chm_path,  INPUT_HEIGHT=wallh_path, INPUT_ASPECT=walla_path,
#                  UTC=0, OUTPUT_DIR=f'{fronti}/ams2_solweig2d_svf', INPUT_MET=met_data, INPUT_DTM=dtm_path, INPUT_EXTRAHEIGHT=6, INPUT_MULT_DSMS=mult_dsms_path, INPUT_LC=landcover_path,  INPUT_DEM=None, INPUT_ANISO=f'{fronti}/2d/shadowmats.npz',
#                  CONIFER_TREES=False, INPUT_THEIGHT=25, INPUT_TDSM=None, TRANS_VEG=15, LEAF_START=97, LEAF_END=300,
#                  USE_LC_BUILD=True, SAVE_BUILD=False, ALBEDO_WALLS=0.2, ALBEDO_GROUND=0.15,
#                  EMIS_WALLS=0.9, EMIS_GROUND=0.95, ABS_S=0.7, ABS_L=0.95, POSTURE=0,  ONLYGLOBAL=True,
#                  OUTPUT_TMRT=True, OUTPUT_LUP=True, OUTPUT_KUP=True, OUTPUT_KDOWN=True, OUTPUT_LDOWN=True, OUTPUT_SH=True, OUTPUT_TREEPLANTER=False,
#                  CYL=True).processAlgorithm()
# fronti =f"{d}:/Geomatics/thesis/_amsterdamset/location"
#
# list = [3, 4, 5]
# schiphol_file = f"{d}:/Geomatics/thesis/_amsterdamset/23aug/aug23_schip_climate_qgis.txt" # f"{d}:/Geomatics/thesis/_amsterdamset/12sep/sep12_schip_climatebike_qgis.txt"
#
# for i in list:
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
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_gap_climate"
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
#     OUTPUT_DIR = f"{d}:/Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_over_climate"
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
#     OUTPUT_DIR = f"{d}:Geomatics/thesis/_amsterdamset/location_{loc}/new/solweig_3d_climate"
#
#     # INPUT_ANISO=INPUT_ANISO
#     test = SOLWEIGAlgorithm(INPUT_DSM, INPUT_SVF, INPUT_CDSM, INPUT_HEIGHT, INPUT_ASPECT, UTC, OUTPUT_DIR,
#                             INPUT_MET, INPUT_LC=INPUT_LC, INPUT_DTM=INPUT_DTM, INPUT_MULT_DSMS=MULT_DSMS,
#                             INPUT_ANISO=INPUT_ANISO)
#     test.processAlgorithm_3d()
