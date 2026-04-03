import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 页面与全局设置
# ==========================================
st.set_page_config(page_title="黑土地作物产量与施肥决策系统", layout="wide", page_icon="🌾")

# 设置中文字体，防止图表中文乱码 (适用于 Windows)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 动态加载 SVR 模型
# ==========================================
# 这里指向您刚才生成 pkl 文件的 SVR_Models 文件夹
MODEL_DIR = "SVR_Models"

@st.cache_resource
def load_model(crop_choice):
    try:
        if crop_choice == "大豆 (Soybean)":
            model_path = os.path.join(MODEL_DIR, "svr_model_Soybean.pkl")
        else:
            model_path = os.path.join(MODEL_DIR, "svr_model_Maize.pkl")
            
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"未找到模型文件: {model_path}。请确认路径是否正确！")
        return None

# ==========================================
# 3. 侧边栏：核心控制选项
# ==========================================
st.sidebar.title("⚙️ 系统控制台")
crop_type = st.sidebar.selectbox("选择目标作物", ["大豆 (Soybean)", "玉米 (Maize)"])

# 根据选择动态加载模型
current_model = load_model(crop_type)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 核心功能选项")
run_prediction = st.sidebar.button("计算预估产量")
run_fertilizer = st.sidebar.button("生成下一季施肥方案")

# ==========================================
# 4. 主界面：参数输入区
# ==========================================
st.title("🌾 东北黑土地产量预测与智能施肥系统")
st.markdown("""
基于支持向量回归 (SVR) 算法构建。请输入当前地块的基础数据，系统将预测本季产量，并通过偏依赖分析提供下一季的最佳施肥策略。
""")

with st.expander("📝 必填项：田间基础参数与气象预期", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌱 土壤本底化验数据**")
        ph = st.number_input("土壤 pH 值", min_value=4.0, max_value=10.0, value=8.5, step=0.1)
        om = st.number_input("有机质 OM (g/kg)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        
    with col2:
        st.markdown("**🧪 当前土壤速效养分**")
        an = st.number_input("速效氮 AN (mg/kg)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
        ap = st.number_input("速效磷 AP (mg/kg)", min_value=0.0, max_value=150.0, value=40.0, step=1.0)
        ak = st.number_input("速效钾 AK (mg/kg)", min_value=0.0, max_value=300.0, value=170.0, step=1.0)
        
    with col3:
        st.markdown("**🌤️ 气象与管理因素**")
        temp = st.number_input("生育期均温 (°C)", min_value=10.0, max_value=30.0, value=19.2, step=0.1)
        precip = st.number_input("生育期降水 (mm)", min_value=10.0, max_value=1000.0, value=30.0, step=1.0)
        density = st.number_input("种植密度 (株/公顷)", min_value=10000, max_value=300000, value=180000 if crop_type == "大豆 (Soybean)" else 60000, step=1000)

with st.expander("🛠️ 可选项：高级辅助参数 (选填)"):
    col4, col5, col6 = st.columns(3)
    with col4:
        tn = st.number_input("全氮 TN (%)", min_value=0.0, max_value=1.0, value=0.16, step=0.01)
    with col5:
        tp = st.number_input("全磷 TP (%)", min_value=0.0, max_value=1.0, value=0.14, step=0.01)
    with col6:
        tk = st.number_input("全钾 TK (%)", min_value=0.0, max_value=5.0, value=2.4, step=0.1)
        rotation = st.selectbox("轮作类型", [0, 1, 2, 3], index=2, help="0:连作, 1-3:不同轮作模式")

# ==========================================
# 5. 后台逻辑：构建输入特征矩阵
# ==========================================
def build_feature_df():
    # 注意：这里的列名必须和 SVR 训练时清洗后的列名完全一致！
    density_col_name = 'Plant number per ha' if crop_type == "大豆 (Soybean)" else 'Crop number per ha'
    
    # 兼容处理：如果数据集中大豆的密度列叫 'Plant number per mu'，请在这里手动修改替换
    features = {
        'Rotation type': [rotation],
        'Temperature': [temp],
        'Precipitation': [precip],
        'pH': [ph],
        'OM': [om],
        'AN': [an],
        'AP': [ap],
        'AK': [ak],
        'TN': [tn],
        'TP': [tp],
        'TK': [tk],
        density_col_name: [density]
    }
    
    # 为了防止模型因为输入列名和训练列名顺序不一致报错，最好转化为 Numpy 数组输入
    # 或者确保这里的字典顺序就是数据库里从左到右的顺序
    return pd.DataFrame(features)

input_df = build_feature_df()

# ==========================================
# 6. 模块一：产量预测
# ==========================================
if run_prediction:
    if current_model is not None:
        with st.spinner('模型正在基于黑土地本底数据计算中...'):
            # 使用 .values 规避特征名称不完全匹配的警告
            pred_yield = current_model.predict(input_df.values)[0]
            
            st.success("计算完成！")
            st.metric(label=f"🏆 预测 {crop_type} 产量", value=f"{pred_yield:.2f} kg/ha")
            st.info("💡 提示：如需探寻提高产量的最佳干预手段，请点击左侧的“生成下一季施肥方案”。")

# ==========================================
# 7. 模块二：下一季施肥指导 (动态模拟)
# ==========================================
if run_fertilizer:
    if current_model is not None:
        st.markdown("### 📊 下一季速效氮 (AN) 追施模拟曲线")
        st.write("系统正在通过保持其他环境和土壤条件不变，模拟不同施肥量下的产量响应曲线...")
        
        # 模拟氮肥(AN)从当前值的 40% 变化到 200%
        an_test_values = np.linspace(an * 0.4, an * 2.0, 50)
        simulated_yields = []
        
        for val in an_test_values:
            temp_df = input_df.copy()
            temp_df['AN'] = val 
            simulated_yields.append(current_model.predict(temp_df.values)[0])
            
        best_idx = np.argmax(simulated_yields)
        best_an = an_test_values[best_idx]
        max_yield = simulated_yields[best_idx]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(an_test_values, simulated_yields, label='产量响应曲线', color='#2ca02c', linewidth=2.5)
        ax.axvline(x=an, color='gray', linestyle='--', label=f'当前水平 ({an:.1f})')
        ax.axvline(x=best_an, color='red', linestyle='--', label=f'理论最优 ({best_an:.1f})')
        ax.scatter([best_an], [max_yield], color='red', s=80, zorder=5)
        
        ax.set_title(f"{crop_type} 氮肥施用量与预测产量的关系", fontsize=14, fontweight='bold')
        ax.set_xlabel("土壤速效氮 AN (mg/kg)", fontsize=12)
        ax.set_ylabel("预测产量 (kg/ha)", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("### 👨‍🌾 农技指导建议")
        if best_an > an * 1.05: # 给 5% 的缓冲容错率
            st.warning(f"**建议增施氮肥**：当前速效氮为 {an} mg/kg，模型显示当土壤速效氮达到 **{best_an:.1f} mg/kg** 时，产量可达峰值 ({max_yield:.1f} kg/ha)。下一季可考虑适当增加底肥或追施氮肥。")
        elif best_an < an * 0.95:
            st.success(f"**建议减量控氮**：当前存在养分过剩或拮抗风险！模型显示最佳速效氮水平约为 **{best_an:.1f} mg/kg**。下一季可减少氮肥投入，降低成本并保护黑土地生态。")
        else:
            st.info("**保持现状**：当前的氮肥水平已经处于模型预测的理想范围内，建议下一季维持现有施肥方案。")