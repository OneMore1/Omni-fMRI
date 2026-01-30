import os
import re
import glob

# ================= 配置区域 =================
# 1. 日志文件所在的目录
LOG_DIR = "/mnt/afs/250010058/neurostorm/neurostorm_ncc/LOGS"

# 2. 想要替换成的字符串
REPLACEMENT_TEXT = "*******"

# 3. 敏感路径的前缀 (正则匹配的核心)
# 这里指定匹配 /mnt/afs 开头的内容
SENSITIVE_PREFIX = "/vePFS-0x0d"
# ===========================================

def clean_logs():
    print(f"🔍 正在扫描目录: {LOG_DIR}")
    
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误: 目录不存在 - {LOG_DIR}")
        return

    # 获取目录下所有的 txt 文件
    log_files = glob.glob(os.path.join(LOG_DIR, "*.json"))
    
    if not log_files:
        print("⚠️  未找到任何 .txt 文件。")
        return

    # 编写正则表达式
    # 解释:
    # /mnt/afs      -> 匹配字面量 /mnt/afs
    # [^\s"':,\])]+ -> 匹配后面紧跟的非空白字符、非引号、非冒号、非逗号等分隔符
    #                  (这样可以保证匹配到完整的路径，但不会把句号或引号匹配进去)
    regex_pattern = re.compile(rf"{re.escape(SENSITIVE_PREFIX)}[^\s\"':,<>\[\]()]*")

    processed_count = 0

    for file_path in log_files:
        file_name = os.path.basename(file_path)
        
        try:
            # 1. 读取原文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 2. 检查是否有匹配项
            matches = regex_pattern.findall(content)
            if not matches:
                continue # 如果没有敏感路径，跳过该文件

            # 3. 执行替换
            # 使用 sub 进行全文替换
            new_content = regex_pattern.sub(REPLACEMENT_TEXT, content)
            
            # 4. 将脱敏后的内容写回文件 (覆盖原文件)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            processed_count += 1
            print(f"✅ 已脱敏: {file_name} (替换了 {len(matches)} 处路径)")

        except Exception as e:
            print(f"❌ 处理文件失败 {file_name}: {e}")

    print(f"\n🎉 处理完成！共修改了 {processed_count} 个日志文件。")

if __name__ == "__main__":
    # 为了防止误操作，建议先在一个测试文件上试一下，或者直接运行：
    clean_logs()