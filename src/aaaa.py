import os
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import shutil

def fix_log_tags(log_dir):
    # 1. 设置标签
    old_tag = 'Episode/02-SR_Last100'
    new_tag = 'Episode/03-SuccessRate_Last100'
    
    # 创建临时输出目录
    tmp_output = os.path.join(log_dir, "tmp_fixed")
    if os.path.exists(tmp_output):
        shutil.rmtree(tmp_output)

    # 2. 加载原始数据
    print(f"正在读取数据: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 3. 创建新的写入器
    writer = SummaryWriter(tmp_output)

    # 遍历所有标量
    scalar_tags = ea.Tags()['scalars']
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        current_tag = new_tag if tag == old_tag else tag
        
        print(f"正在处理标签: {tag} -> {current_tag}")
        for event in events:
            # 修复点：移除 wall_time 参数，或者使用 positional argument
            # 大多数版本的 add_scalar 签名是 (tag, scalar_value, global_step)
            writer.add_scalar(current_tag, event.value, event.step)
            
    writer.close()
    print(f"数据重写完成，临时存放在: {tmp_output}")

    # 4. 替换原始文件
    backup_dir = log_dir + "_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(log_dir, backup_dir)
        print(f"已备份原始日志至: {backup_dir}")

    # 清理原目录并移入新文件
    for item in os.listdir(log_dir):
        item_path = os.path.join(log_dir, item)
        if item == "tmp_fixed": continue
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # 移动文件
    for item in os.listdir(tmp_output):
        shutil.move(os.path.join(tmp_output, item), log_dir)
    
    os.rmdir(tmp_output)
    print(">>> [成功] 标签名称已修改。请重启 TensorBoard 查看。")

if __name__ == "__main__":
    # 请确保此路径相对于你运行脚本的位置是正确的
    target_path = "src/scripts/outputs/ENV1/KFDQN_seed123_20260113_151044/logs"
    
    # 如果是相对路径，建议先转换成绝对路径
    abs_path = os.path.abspath(target_path)
    
    if os.path.exists(abs_path):
        fix_log_tags(abs_path)
    else:
        print(f"错误：找不到路径 {abs_path}")