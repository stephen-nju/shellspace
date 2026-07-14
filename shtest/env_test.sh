# #!/usr/bin/bash

# change default dash to bash
# source /opt/conda/etc/profile.d/conda.sh
# conda activate zb_swift && which swift

# export enable_thinking=true

# if [[ $enable_thinking = true ]]; then
# 	echo "enable_thinking is true"
# fi

#!/bin/bash

# is_true() {
# 	local val="${1,,}" # 转换为全小写以忽略大小写差异
# 	case "$val" in
# 	true | 1) return 0 ;;  # 返回0表示true（UNIX标准成功状态）
# 	false | 0) return 1 ;; # 返回非0表示false
# 	*)
# 		if [[ "$val" =~ ^[[:blank:]]*(true|false|1|0)[[:blank:]]*$ ]]; then
# 			# 处理带有空白字符的特殊情况（可选增强）
# 			local stripped_val=$(echo "$val" | xargs) # 去除前后空白
# 			case "$stripped_val" in
# 			true | 1) return 0 ;;
# 			*) return 1 ;;
# 			esac
# 		else
# 			return 2 # 额外错误码标识非法输入（扩展能力）
# 		fi
# 		;;
# 	esac
# }


# export data=$(is_true "yes")
# echo $data


cat << EOF >> ~/.zshrc
PROXY=http://proxysys.his.hihonor.com:8080/
export http_proxy=\${PROXY}
export https_proxy=\${PROXY}
export HTTP_PROXY=\${PROXY}
export HTTPS_PROXY=\${PROXY}
export no_proxy='localhost,127.0.0.1,w3.hihonor.com,mgit-tm.ipd.hihonor.com,mirrors.chinatelecom.hihonor.io,aptps.cloud.hihonor.com,gitlab-y.ipd.hihonor.com'
EOF
