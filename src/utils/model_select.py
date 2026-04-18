def model_select(loss_type):
    if loss_type == "proxybased":
        from src.model.hist_model.model_proxybased import Model_Proxybased as MyModel
    elif loss_type == 'acnet':
        from src.model.hist_model.model_ACNet import ACNet as MyModel
    elif loss_type == 'multiview':
        from src.model.hist_model.model_multiview import Model_MultiView as MyModel
    elif loss_type == "2DTDS":
        from src.model.hist_model.model_2DTDS import Model_2DTDS as MyModel
    elif loss_type == "3DTDS":
        from src.model.hist_model.model_3DTDS import Model_3DTDS as MyModel
    elif loss_type == "gait_set":
        from src.model.hist_model.model_gaitset import Model_Gaitset as MyModel
    elif loss_type == "gait_set_pan":
        from src.model.hist_model.model_gaitset_PAN import Model_Gaitset_PAN as MyModel
    elif loss_type == "tdn_21":
        from src.model.hist_model.model_tdn_2021 import Model_Tdn as MyModel
    elif loss_type == "tdn_18":
        from src.model.hist_model.model_tdn_2018 import Model_Tdn as MyModel
    elif loss_type == "lrcn":
        from src.model.hist_model.model_LRCN import Model_LRCN as MyModel
    elif loss_type == "gru_rcn":
        from src.model.hist_model.model_gru_rcn import Model_GRU_RCN as MyModel
    elif loss_type == "trn":
        from src.model.hist_model.model_TRN import Model_TRN as MyModel
    elif loss_type == "tpn":
        from src.model.hist_model.model_TPN import Model_TPN as MyModel
    elif loss_type == "tan":
        from src.model.hist_model.model_TANet import Model_TANet as MyModel
    elif loss_type == "csn":
        from src.model.hist_model.model_CSN import Model_CSN as MyModel
    elif loss_type == "stnet":
        from src.model.hist_model.model_StNet import Model_StNet as MyModel
    elif loss_type == "diflow":
        from src.model.hist_model.model_diflow import Model_DiFlow as MyModel
    elif loss_type == "diblock":
        from src.model.hist_model.model_diblock import Model_DiBlock as MyModel
    elif loss_type == "pb_net":
        from src.model.hist_model.PBNet import Model_PBNet as MyModel
    elif loss_type == "rd_net":
        from src.model.hist_model.CMLGNet.RDNet import Model_RDNet as MyModel
    elif loss_type == 'rd_net_mem':
        from src.model.RDNet_mem import Model_RDNet as MyModel
    elif loss_type == "rd_net_pyra":
        from src.model.hist_model.CMLGNet.RDNet_pyra import Model_RDNet as MyModel
    elif loss_type == "rd_net_cmt":
        from src.model.hist_model.CMLGNet.RDNet_cmt import Model_RDNet as MyModel
    elif loss_type == "TSN":
        from src.model.hist_model.model_TSN import Model_TSN as MyModel
    elif loss_type == "tsn_pyra":
        from src.model.TSN_pyra import Model_TSN as MyModel
    elif loss_type == "am_net":
        from src.model.model_AMNet import Model_AMNet as MyModel
    elif loss_type == "tam":
        from src.model.hist_model.model_TAM import Model_Tam as MyModel
    elif loss_type == "dscmt":
        from src.model.hist_model.DSCMT import Model_DSCMT as MyModel
    elif loss_type == "unet":
        from src.model.UNet import UNet as MyModel
    elif loss_type == "unet_temp":
        from src.model.UNet_temp import UNet as MyModel
    elif loss_type == "unet_2stage":
        from src.model.UNet_2stage import UNet as MyModel
    elif loss_type == "cmcb":
        from src.model.hist_model.CMCB import Model_CMCB as MyModel
    elif loss_type == "tscnn":
        from src.model.hist_model.TSCNN import Model_TSCNN as MyModel
    elif loss_type == "cscmft":
        from src.model.hist_model.CSCMFT import Model_CSCMFT as MyModel
    elif loss_type == "cmlg":
        from src.model.hist_model.CMLGNet import Model_RDNet as MyModel
    else:
        print("loss_type unknown")
        exit(0)

    return MyModel