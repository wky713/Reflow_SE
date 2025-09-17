import os
import glob
import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm
from flowmse.model import VFModel
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    # 固定参数配置
    class Args:
        test_dir = "/home/wukaiying0713/flowmse_re/flowmse/data"#带噪语音文件位置
        ckpt = "/home/wukaiying0713/flowmse_re/flowmse/ckpt/epoch=208_last.ckpt"
        odesolver_type = "white"
        odesolver = "euler"
        N = 30
        reverse_starting_point = 1.0
        last_eval_point = 0.03
    
    args = Args()
    
    # 输入输出路径设置
    noisy_dir = os.path.join(args.test_dir, "train", "noisy")
    output_clean_dir = "/home/wukaiying0713/flowmse_re/flowmse/data2/train/clean"
    output_noisy_dir = "/home/wukaiying0713/flowmse_re/flowmse/data2/train/noisy"
    
    ensure_dir(output_clean_dir)
    ensure_dir(output_noisy_dir)

    # 加载模型
    model = VFModel.load_from_checkpoint(
        args.ckpt, 
        base_dir="",
        batch_size=4,  # 单样本处理
        num_workers=1,
        kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()

    # 获取带噪语音文件列表
    noisy_files = sorted(glob.glob(f'{noisy_dir}/*.wav'))
    
    # 生成处理循环
    for noisy_file in tqdm(noisy_files):
        filename = os.path.basename(noisy_file)
        
        # 1. 读取带噪语音
        y, sr = load(noisy_file)
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        # 2. 生成干净语音
        with torch.no_grad():
            Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
            Y = pad_spec(Y)
            
            sampler = get_white_box_solver(
                args.odesolver, 
                model.ode, 
                model, 
                Y=Y.cuda(), 
                Y_prior=Y.cuda(),
                T_rev=args.reverse_starting_point,
                t_eps=args.last_eval_point,
                N=args.N
            )
            
            sample, _ = sampler()
            sample = sample.squeeze()
            x_hat = model.to_audio(sample, T_orig)
            x_hat = x_hat * norm_factor
            x_hat = x_hat.squeeze().cpu().numpy()

        # 3. 保存结果
        # 保存生成的干净语音
        write(os.path.join(output_clean_dir, filename), x_hat, sr)
        # 复制带噪语音到新位置
        write(os.path.join(output_noisy_dir, filename), y.squeeze().numpy(), sr)

if __name__ == '__main__':
    main()