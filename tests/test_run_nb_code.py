import pytest

from agentica.tools.run_nb_code import RunNbCodeWrapper


def test_code_running():
    executor = RunNbCodeWrapper()
    output, is_success = executor.run("print('hello world!')")
    assert is_success
    executor.terminate()


def test_split_code_running():
    executor = RunNbCodeWrapper()
    _ = executor.run("x=1\ny=2")
    _ = executor.run("z=x+y")
    output, is_success = executor.run("assert z==3")
    assert is_success
    executor.terminate()


def test_execute_error():
    executor = RunNbCodeWrapper()
    output, is_success = executor.run("z=1/0")
    assert not is_success
    executor.terminate()


PLOT_CODE = """
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
random_data = np.random.randn(1000)  # 生成1000个符合标准正态分布的随机数

# 绘制直方图
plt.hist(random_data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

# 添加标题和标签
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图形
plt.show()
plt.close()
"""


def test_plotting_code():
    executor = RunNbCodeWrapper()
    output, is_success = executor.run(PLOT_CODE)
    assert is_success
    executor.terminate()


def test_run_with_timeout():
    executor = RunNbCodeWrapper(timeout=1)
    code = "import time; time.sleep(2)"
    message, success = executor.run(code)
    assert not success
    assert message.startswith("Cell execution timed out")
    executor.terminate()


def test_run_code_text():
    executor = RunNbCodeWrapper()
    message, success = executor.run(code='print("This is a code!")', language="python")
    assert success
    assert "This is a code!" in message
    message, success = executor.run(code="# This is a code!", language="markdown")
    assert success
    assert message == "# This is a code!"
    mix_text = "# Title!\n ```python\n print('This is a code!')```"
    message, success = executor.run(code=mix_text, language="markdown")
    assert success
    assert message == mix_text
    executor.terminate()


@pytest.mark.parametrize(
    "k", [(1), (5)]
)  # k=1 to test a single regular terminate, k>1 to test terminate under continuous run
def test_terminate(k):
    for _ in range(k):
        executor = RunNbCodeWrapper()
        executor.run(code='print("This is a code!")', language="python")
        is_kernel_alive = executor.nb_client.km.is_alive()
        assert is_kernel_alive
        executor.terminate()
        assert executor.nb_client.km is None
        assert executor.nb_client.kc is None


def test_reset():
    executor = RunNbCodeWrapper()
    executor.run(code='print("This is a code!")', language="python")
    is_kernel_alive = executor.nb_client.km.is_alive()
    assert is_kernel_alive
    executor.reset()
    assert executor.nb_client.km is None
    executor.terminate()


def test_parse_outputs():
    executor = RunNbCodeWrapper()
    code = """
    import pandas as pd
    df = pd.DataFrame({'ID': [1,2,3], 'NAME': ['a', 'b', 'c']})
    print(df.columns)
    print(f"columns num:{len(df.columns)}")
    print(df['DUMMPY_ID'])
    """
    output, is_success = executor.run(code)
    print('output:', output)
    assert not is_success
    executor.terminate()
