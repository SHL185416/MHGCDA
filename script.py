import os

citation1_acmv9_parameter = " --PP_wei=1 --Clf_wei=1 --NN_wei=1 --P_wei=1 --attn_drop=0.3 --batch_size=4000" \
                            " --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.6 --l2_w=0.01 --lr_ini=0.01" \
                            " --num_heads=16 --num_hidden=16 --num_layers=2 --num_out_heads=3 --target='citation1_acmv9'" \
                            " --tau_n_rate=0.2 --tau_p_rate=0.8"
citation1_citationv1_parameter = " --PP_wei=1 --Clf_wei=1 --NN_wei=1 --P_wei=1 --attn_drop=0.7 --batch_size=6000" \
                                 " --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.3 --l2_w=0.01 --lr_ini=0.01" \
                                 " --num_heads=16 --num_hidden=16 --num_layers=2 --num_out_heads=3 --target='citation1_citationv1'" \
                                 " --tau_n_rate=0.2 --tau_p_rate=0.8"
citation1_dblpv7_parameter = " --PP_wei=1 --Clf_wei=1 --NN_wei=1 --P_wei=1 --attn_drop=0.7 --batch_size=5000" \
                             " --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.4 --l2_w=0.01 --lr_ini=0.01" \
                             " --num_heads=16 --num_hidden=16 --num_layers=2 --num_out_heads=2 --target='citation1_dblpv7'" \
                             " --tau_n_rate=0.2 --tau_p_rate=0.8"

citation2_acmv8_parameter = " --PP_wei=1 --Clf_wei=1 --NN_wei=1 --P_wei=0.001 --attn_drop=0.3 --batch_size=5000" \
                            " --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.3 --l2_w=0.001 --lr_ini=0.01" \
                            " --num_heads=32 --num_hidden=8 --num_layers=3 --num_out_heads=1 --target='citation2_acmv8'" \
                            " --tau_n_rate=0.1 --tau_p_rate=0.9"
citation2_citationv1_parameter = " --PP_wei=0.1 --Clf_wei=1 --NN_wei=1 --P_wei=0.1 --attn_drop=0.5 --batch_size=2000" \
                                 " --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.3 --l2_w=0.0001 --lr_ini=0.01" \
                                 " --num_heads=32 --num_hidden=16 --num_layers=3 --num_out_heads=1 --target='citation2_citationv1'" \
                                 " --tau_n_rate=0.2 --tau_p_rate=0.9"
citation2_dblpv4_parameter = " --PP_wei=1 --Clf_wei=1 --NN_wei=1 --P_wei=1 --attn_drop=0.1 --batch_size=4000 " \
                             " --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.3 --l2_w=0.001 --lr_ini=0.001" \
                             " --num_heads=32 --num_hidden=16 --num_layers=1 --num_out_heads=2 --target='citation2_dblpv4'" \
                             " --tau_n_rate=0.3 --tau_p_rate=0.8"

# [citation2_acmv8_parameter, citation2_citationv1_parameter, citation2_dblpv4_parameter,
# citation1_acmv9_parameter, citation1_citationv1_parameter, citation1_dblpv7_parameter]
parameter_list = [citation1_citationv1_parameter]

for parameter in parameter_list:
    cmd = f"python main.py" + parameter
    os.system(cmd)
