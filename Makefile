SHELL := /bin/bash
ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: help doctor setup patches verify serve status stop kill watchdog-start watchdog-stop watchdog-status router-restart smoke bench bench8 test lint clean

help:
	@echo "vllm-ascend-deepseek-v41-a3"
	@echo
	@echo "  make setup           解包 rootfs（下载 OCI 镜像层）"
	@echo "  make patches         应用优化补丁（幂等）"
	@echo "  make verify          校验补丁与 6 个 gate"
	@echo "  make serve           启动服务（后台）"
	@echo "  make status          服务状态"
	@echo "  make stop            停止服务（含残留清理）"
	@echo "  make kill            停止 watchdog + 服务 + NPU worker"
	@echo "  make watchdog-start  启动 watchdog（生产推荐）"
	@echo "  make watchdog-status watchdog 状态"
	@echo "  make watchdog-stop   停止 watchdog（不影响服务）"
	@echo "  make router-restart  只重启 prefix router（不动 vLLM）"
	@echo "  make doctor          只读预检（rootfs/权重/补丁/端口）"
	@echo "  make smoke           冒烟测试"
	@echo "  make bench           2K 吞吐压测"
	@echo "  make bench8          8K 推测解码压测"
	@echo "  make test            单元测试（patch 状态 / OCI 解包）"
	@echo "  make lint            bash -n / py_compile"

doctor:
	bash $(ROOT)/scripts/doctor.sh

setup:
	bash $(ROOT)/scripts/setup_rootfs.sh

patches:
	bash $(ROOT)/scripts/apply_patches.sh apply

verify:
	bash $(ROOT)/scripts/apply_patches.sh status

serve:
	bash $(ROOT)/scripts/serve.sh

status:
	STATUS=1 bash $(ROOT)/scripts/serve.sh

stop:
	STOP=1 bash $(ROOT)/scripts/serve.sh

kill:
	bash $(ROOT)/scripts/kill_vllm.sh

watchdog-start:
	bash $(ROOT)/scripts/watchdog.sh start

watchdog-status:
	bash $(ROOT)/scripts/watchdog.sh status

watchdog-stop:
	bash $(ROOT)/scripts/watchdog.sh stop

router-restart:
	ROUTER_RESTART=1 bash $(ROOT)/scripts/serve.sh

smoke:
	python3 $(ROOT)/tests/smoke_test.py

bench:
	python3 $(ROOT)/scripts/bench_2k.py

bench8:
	python3 $(ROOT)/scripts/bench_8k.py

test:
	python3 -m unittest discover -s $(ROOT)/tests -v

lint:
	@for f in $(ROOT)/scripts/*.sh $(ROOT)/examples/*.sh; do bash -n "$$f" || exit 1; done
	@python3 -m py_compile $(ROOT)/scripts/*.py $(ROOT)/tests/*.py $(ROOT)/examples/*.py
	@command -v shellcheck >/dev/null 2>&1 && shellcheck -S warning $(ROOT)/scripts/*.sh $(ROOT)/examples/*.sh || true
	@echo "lint ok"

clean:
	rm -rf $(ROOT)/logs/*.log
	find $(ROOT) -type d -name __pycache__ -prune -exec rm -rf {} +
