"use strict";

const translations = {
  en: {
    "crystal.multi": "Multicomponent",
    "crystal.multi.full": "Multicomponent crystal",
    "crystal.ionic": "Ionic",
    "crystal.ionic.full": "Ionic crystal",
    "crystal.enlarge": "Enlarge crystal",
    "crystal.alt.template": "SALA-generated {type}, target {id}, shown as a fully relaxed 2 × 2 × 2 supercell in tubular style. Generation symmetry {symmetry}; hydrogen atoms omitted.",
    "gallery.types": "Chemical type",
    "gallery.all": "All types",
    "gallery.system": "Generation crystal system",
    "gallery.system.all": "All crystal systems",
    "gallery.count": "{shown} of {total} cases",
    "gallery.empty": "No cases match this combination.",
    "gallery.reset": "Show all cases",
    "system.triclinic": "Triclinic",
    "system.monoclinic": "Monoclinic",
    "system.orthorhombic": "Orthorhombic",
    "system.tetragonal": "Tetragonal",
    "system.trigonal": "Trigonal",
    "system.hexagonal": "Hexagonal",
    "meta.title": "SALA — Complex molecules. Ordered crystals.",
    "meta.description": "SALA generates complex molecular crystals through symmetry-constrained state–context separation. Explore the framework, generated structures and research results.",
    skip: "Skip to content", "nav.label": "Main navigation", "nav.structures": "Structures", "nav.cases": "Cases", "nav.framework": "Framework", "nav.results": "Results", "nav.figures": "Figures", "nav.status": "Status", "nav.open": "Open navigation", "nav.close": "Close navigation",
    "rail.label": "Page index", "rail.cover": "Cover",
    "head.left": "SALA · Molecular crystals", "head.center": "Symmetry-constrained generation", "head.subject": "Molecular crystal generation",
    "cover.tag1": "SALA", "cover.tag2": "Molecular crystal generation", "cover.scroll": "Scroll",
    "colophon.note": "Model architecture code is released under the MIT License; checkpoints remain unreleased.",
    "subject.structures": "Generated structures", "subject.cases": "Selected cases", "subject.framework": "The framework", "subject.results": "Research results", "subject.figures": "Manuscript figures", "subject.status": "Research status",
    "spec.rendering": "Rendering", "spec.rendering.value": "Tubular style · custom element colours",
    "spec.cell": "Cell", "spec.cell.value": "2 × 2 × 2 supercell · fully relaxed",
    "spec.atoms": "Atoms", "spec.atoms.value": "Heavy atoms only · hydrogen omitted",
    "cases.eyebrow": "SELECTED CASES", "cases.title": "Nine generated crystals.", "cases.description": "Filter by chemical type and generation crystal system. Every card opens an enlarged rotation with its own download.",
    "hero.description": "Molecular conformation, lattice geometry, and crystal packing. Generated together, guided by symmetry.", "hero.visual": "GENERATED CRYSTAL", "hero.supercell": "SUPERCELL",
    "motion.pause": "Pause rotations", "motion.play": "Play rotations", "crystal.previous": "Previous crystal", "crystal.next": "Next crystal", "crystal.legend.label": "Custom element colours", "crystal.legend.title": "Tubular style · custom element colours", "crystal.choose": "Choose a crystal", "crystal.symmetry": "Generation symmetry", "crystal.single": "Single-component", "crystal.metal": "Metal-containing", "crystal.single.full": "Single-component crystal", "crystal.metal.full": "Metal-containing crystal", "crystal.modal.note": "2 × 2 × 2 supercell · Tubular rendering · Custom element colours · Fully relaxed · Hydrogen atoms omitted. Space group refers to the generation condition before relaxation.",
    "structures.eyebrow": "GENERATED STRUCTURES", "structures.title": "A closer look at crystalline order.", "structures.description": "Selected SALA-generated structures across four chemical types and six crystal systems. Explore the different packing arrangements in three dimensions.", "structures.note": "2 × 2 × 2 supercells · Fully relaxed · Hydrogen atoms omitted. *Crystal systems and space groups refer to the generation condition, before relaxation.",
    "framework.eyebrow": "THE FRAMEWORK", "framework.line1": "A compact state.", "framework.line2": "A complete crystal context.", "framework.description": "Separate what needs to evolve from what the model needs to see. Symmetry connects the two.", "framework.expansion": "Symmetry-Aware Lattice and Asymmetric-unit generation", "framework.panel.title": "SALA's approach",
    "challenge.title": "Three quantities, one decision", "challenge.lede": "Molecular crystal generation must coordinate conformation, lattice geometry and periodic packing.",
    "challenge.one.title": "Conformation", "challenge.one.body": "The heavy-atom geometry of the molecules inside the asymmetric unit.",
    "challenge.two.title": "Lattice geometry", "challenge.two.body": "Cell parameters that remain compatible with the target symmetry.",
    "challenge.three.title": "Periodic packing", "challenge.three.body": "How symmetry copies and neighbouring cells arrange in the full periodic environment.",
    "method.state.title": "Evolve the independent state", "method.state.body": "Generate only the asymmetric-unit coordinates and symmetry-compatible lattice parameters.", "method.context.title": "Reason over the full context", "method.context.body": "Expand the full cell and model periodic interactions with a structure-aware transformer.", "method.update.title": "Return to symmetry", "method.update.body": "Map coordinated predictions back to the independent state through OrbitPullback.",
    "scope.classes": "Chemical classes", "scope.classes.note": "Single-component, multicomponent, ionic, metal-containing",
    "scope.systems": "Crystal systems", "scope.systems.note": "All seven represented in the held-out benchmark",
    "scope.candidates": "Candidates per target", "scope.candidates.note": "SALA budget; the all-atom method baseline uses 50–3,150",
    "stat.targets": "Held-out crystals", "stat.hits": "Archived packing-threshold hits", "stat.packing": "Mean target-best packing RMSD", "stat.conformer": "Mean target-best conformer RMSD",
    "figures.eyebrow": "MANUSCRIPT FIGURES", "figures.title": "The original evidence.", "figures.description": "Four original manuscript figures. Each one carries its own reading note.", "figures.reading": "How to read it",
    "figure.enlarge": "Enlarge figure", "figure.framework.open": "Enlarge SALA framework figure", "figure.results.open": "Enlarge research figure", "figure.tabs": "Research figures",
    "figure.framework.title": "The SALA framework", "figure.framework.caption": "SALA separates a symmetry-constrained generative state from full-cell context.", "figure.framework.alt": "SALA framework showing reduced atomic and lattice degrees of freedom, lattice masks, asymmetric-unit expansion, periodic context, the transformer and orbit pullback.",
    "figure.performance.title": "Reference recovery", "figure.performance.caption": "Recovery of molecular crystals across chemical classes and crystal systems.", "figure.performance.alt": "Generated/reference crystal overlays, five structural metrics, and recovery rates by crystal system and chemical class for SALA and all-atom method baseline.", "figure.performance.context": "Figure 2 · Original manuscript figure (the all-atom method baseline is labelled CLARI in the original). Recovery panels use archived packing-threshold hits, without joint clash screening. SALA uses 50 candidates per target; all-atom method baseline uses target-specific pools of 50–3,150. Continuous error metrics are minimized independently within each pool. Lower is better for errors; SALA does not lead every metric (relative volume error: SALA 0.40%, all-atom method baseline 0.14%).",
    "figure.complexity.title": "Complexity", "figure.complexity.caption": "SALA's packing advantage widens with molecular and crystallographic complexity.", "figure.complexity.alt": "Packing RMSD stratified by asymmetric-unit size, full-cell size, molecular multiplicity and symmetry operations, with paired-error heatmaps.", "figure.complexity.context": "Figure 3 · Target-best packing errors are grouped by structural complexity. Results describe the evaluated benchmark and do not imply universal success across all space groups.",
    "figure.dynamics.title": "Generation dynamics", "figure.dynamics.caption": "Trajectory-resolved structural organization and attention redistribution.", "figure.dynamics.alt": "Structural progress, pair-class attention enrichment, distance-time attention maps and persistent-contact analysis across SALA generation trajectories.", "figure.dynamics.context": "Figure 4 · These analyses show correspondence between structural organization and attention during generation; they do not establish causal attribution to individual model components.",
    "figure.ablation.title": "Ablations", "figure.ablation.caption": "Ablations of SALA's state–context framework and structure-aware design.", "figure.ablation.alt": "Full SALA and seven separately trained variants compared across conformation, volume, lattice shape, packing and severe overlaps.", "figure.ablation.context": "Figure 5 · Ablations use a common evaluable subset. The full-model packing RMSD here is 1.64 Å; the 1.70 Å benchmark value refers to the complete 799-crystal test set.",
    "results.eyebrow": "RESEARCH RESULTS", "results.title": "Crystal generation, put to the test.", "results.description": "Evaluated against experimental reference crystals across molecular compositions, sizes, and symmetries.", "results.note": "Archived hit rates use packing RMSD below 2 Å only; joint clash-free recovery has not been recomputed. Packing RMSD values are mean target-best errors on the 799-crystal benchmark, evaluated on raw generated heavy-atom structures.",
    "benchmark.eyebrow": "ARCHIVED PACKING-THRESHOLD HITS", "benchmark.summary": "575 of 799 targets have a candidate below 2 Å packing RMSD, using 50 SALA candidates per target.", "benchmark.link": "Explore the evidence", "benchmark.choose": "Choose a benchmark metric", "benchmark.recovery": "Packing hits ↑", "benchmark.rmsd": "Packing RMSD ↓", "benchmark.budgets": "Candidates per target: SALA 50 · all-atom method baseline 50–3,150", "benchmark.unit.recovery": "Higher is better · %", "benchmark.unit.rmsd": "Lower is better · Å", "benchmark.baseline": "All-atom method baseline",
    "paper.authorship": "Authors and affiliation", "paper.affiliation": "Peking University", "paper.title": "Complex molecular crystal generation through symmetry-constrained state–context separation", "paper.title.first": "Complex molecular crystal generation", "paper.title.second": "through symmetry-constrained state–context separation", "paper.abstract.label": "Research overview", "paper.abstract.body": "Molecular crystal generation must coordinate conformation, lattice geometry and periodic packing. SALA uses flow matching to evolve a compact, symmetry-constrained asymmetric-unit state while reasoning over the full periodic environment. This state–context separation enables joint generation across four chemical classes and all seven crystal systems represented in a 799-crystal held-out benchmark.",
    "status.title": "Research status", "status.lede": "This page presents the framework, generated structures and manuscript results. The links below open once their public destinations exist.",
    "links.note": "Architecture code is available. Checkpoints are coming soon.", "links.label": "Research links", "links.paper": "Paper", "links.code": "Code", "links.pending": "Link to be added", "links.checkpoint.pending": "Checkpoint · link to be added", "links.open": "Open resource", "links.checkpoint.open": "Model checkpoints", "links.ready": "Available", "crystal.download": "Download HD GIF",
    "texture.word": "Symmetry",
    "footer.tagline": "Symmetry in the state. Complexity in the context.", "footer.status": "SALA · Molecular crystal generation", "footer.top": "Back to top", "dialog.zoom": "Actual size", "dialog.fit": "Fit to screen", "dialog.close": "Close enlarged view"
  },
  zh: {
    "crystal.multi": "多组分晶体",
    "crystal.multi.full": "多组分晶体",
    "crystal.ionic": "离子型晶体",
    "crystal.ionic.full": "离子型晶体",
    "crystal.enlarge": "放大晶体",
    "crystal.alt.template": "SALA 生成的{type}，目标 {id}，以管式风格展示完整松弛后的 2 × 2 × 2 超胞。生成对称性为 {symmetry}，省略氢原子。",
    "gallery.types": "化学类型",
    "gallery.all": "全部类型",
    "gallery.system": "生成晶系",
    "gallery.system.all": "全部晶系",
    "gallery.count": "显示 {shown} / {total} 个案例",
    "gallery.empty": "没有同时符合这两个条件的案例。",
    "gallery.reset": "显示全部案例",
    "system.triclinic": "三斜晶系",
    "system.monoclinic": "单斜晶系",
    "system.orthorhombic": "正交晶系",
    "system.tetragonal": "四方晶系",
    "system.trigonal": "三方晶系",
    "system.hexagonal": "六方晶系",
    "meta.title": "SALA — 复杂分子，有序晶体。",
    "meta.description": "SALA 通过对称约束的状态–上下文分离生成复杂分子晶体。探索方法框架、真实生成结构和论文研究结果。",
    skip: "跳转到正文", "nav.label": "主导航", "nav.structures": "生成结构", "nav.cases": "案例", "nav.framework": "方法框架", "nav.results": "研究结果", "nav.figures": "论文图", "nav.status": "研究状态", "nav.open": "打开导航", "nav.close": "关闭导航",
    "rail.label": "页码索引", "rail.cover": "封面",
    "head.left": "SALA · 分子晶体", "head.center": "对称约束下的生成", "head.subject": "分子晶体生成",
    "cover.tag1": "SALA", "cover.tag2": "分子晶体生成", "cover.scroll": "向下滑动",
    "colophon.note": "模型架构代码已按 MIT 协议开放，检查点尚未发布。",
    "subject.structures": "生成结构", "subject.cases": "精选案例", "subject.framework": "方法框架", "subject.results": "研究结果", "subject.figures": "论文原图", "subject.status": "研究状态",
    "spec.rendering": "渲染", "spec.rendering.value": "管式风格 · 自定义元素配色",
    "spec.cell": "晶胞", "spec.cell.value": "2 × 2 × 2 超胞 · 完整松弛后结构",
    "spec.atoms": "原子", "spec.atoms.value": "仅重原子 · 省略氢原子",
    "cases.eyebrow": "精选案例", "cases.title": "九个生成晶体。", "cases.description": "按化学类型与生成晶系筛选。点击任意卡片可放大查看旋转动画，并下载高清 GIF。",
    "hero.description": "以对称性为约束，联合生成分子构象、晶格几何与晶体堆积，探索分子在周期空间中的有序组织。", "hero.visual": "生成晶体展示", "hero.supercell": "超胞",
    "motion.pause": "暂停旋转", "motion.play": "播放旋转", "crystal.previous": "上一个晶体", "crystal.next": "下一个晶体", "crystal.legend.label": "自定义元素配色", "crystal.legend.title": "管式风格 · 自定义元素配色", "crystal.choose": "选择晶体", "crystal.symmetry": "生成对称性", "crystal.single": "单组分晶体", "crystal.metal": "含金属晶体", "crystal.single.full": "单组分分子晶体", "crystal.metal.full": "含金属分子晶体", "crystal.modal.note": "2 × 2 × 2 超胞 · 管式渲染 · 自定义元素配色 · 完整松弛后结构 · 省略氢原子。空间群为松弛前的生成条件。",
    "structures.eyebrow": "生成结构", "structures.title": "从每个角度，看见晶体的有序。", "structures.description": "来自四类化学体系、六种晶系的 SALA 生成结构。选择化学类型或晶系，观察不同的三维堆积。", "structures.note": "2 × 2 × 2 超胞 · 完整松弛后结构 · 省略氢原子。*晶系与空间群指松弛前的生成条件。",
    "framework.eyebrow": "方法框架", "framework.line1": "紧凑的生成状态，", "framework.line2": "完整的晶体上下文。", "framework.description": "将需要演化的自由度，与模型需要感知的环境分离，再以对称性连接二者。", "framework.expansion": "对称性感知的晶格与不对称单元生成", "framework.panel.title": "SALA 的方法",
    "challenge.title": "三个自由度，一次联合决策", "challenge.lede": "分子晶体生成需要协同决定分子构象、晶格几何与周期堆积。",
    "challenge.one.title": "分子构象", "challenge.one.body": "不对称单元内分子的重原子几何结构。",
    "challenge.two.title": "晶格几何", "challenge.two.body": "与目标对称性保持相容的晶格参数。",
    "challenge.three.title": "周期堆积", "challenge.three.body": "对称等价副本与相邻晶胞在完整周期环境中的排布方式。",
    "method.state.title": "演化独立状态", "method.state.body": "仅生成不对称单元的坐标，以及与对称性相容的晶格参数。", "method.context.title": "感知完整环境", "method.context.body": "展开完整晶胞，通过结构感知 Transformer 建模周期相互作用。", "method.update.title": "回归对称约束", "method.update.body": "使用 OrbitPullback，将协同预测映射回独立的不对称单元状态。",
    "scope.classes": "化学体系", "scope.classes.note": "单组分、多组分、离子型、含金属",
    "scope.systems": "晶系", "scope.systems.note": "留出测试集中包含全部七大晶系",
    "scope.candidates": "每目标候选数", "scope.candidates.note": "SALA 的候选预算；全原子方法baseline 为 50–3,150",
    "stat.targets": "留出测试晶体", "stat.hits": "归档堆积阈值命中率", "stat.packing": "各目标最优堆积 RMSD 均值", "stat.conformer": "各目标最优构象 RMSD 均值",
    "figures.eyebrow": "论文原图", "figures.title": "原始研究证据。", "figures.description": "四张论文原图。每张都附有对应的解读说明。", "figures.reading": "如何解读",
    "figure.enlarge": "放大查看", "figure.framework.open": "放大 SALA 方法框架图", "figure.results.open": "放大研究结果图", "figure.tabs": "论文结果图",
    "figure.framework.title": "SALA 方法框架", "figure.framework.caption": "SALA 将对称约束的生成状态与完整晶胞上下文分离。", "figure.framework.alt": "SALA 方法框架，展示原子与晶格自由度缩减、晶格约束、ASU 展开、周期上下文、Transformer 与轨道回拉。",
    "figure.performance.title": "参考结构恢复", "figure.performance.caption": "跨化学类别与晶系的分子晶体恢复结果。", "figure.performance.alt": "生成结构与参考结构叠加、五项结构指标，以及 SALA 和 全原子方法baseline 按晶系与化学类别统计的恢复率。", "figure.performance.context": "图 2 · 论文原图（全原子方法baseline 在原图中标作 CLARI）。恢复率面板为按堆积阈值统计的归档命中率，未联合碰撞筛选。SALA 为每个目标生成 50 个候选；全原子方法baseline 按目标使用 50–3,150 个候选。各连续误差指标在候选池中分别取最优值。误差越低越好；SALA 并非在每项指标上均占优（相对体积误差：SALA 0.40%，全原子方法baseline 0.14%）。",
    "figure.complexity.title": "复杂度扩展", "figure.complexity.caption": "随分子与晶体学复杂度增加，SALA 的堆积优势扩大。", "figure.complexity.alt": "按 ASU 大小、完整晶胞大小、分子组分副本数与对称操作数分组的堆积 RMSD，以及配对误差差值热图。", "figure.complexity.context": "图 3 · 按结构复杂度分组比较各目标的最优堆积误差。结论适用于已评估的测试集，并不代表对所有空间群都能普遍成功。",
    "figure.dynamics.title": "生成动力学", "figure.dynamics.caption": "生成轨迹中的结构组织过程与注意力重分配。", "figure.dynamics.alt": "SALA 生成轨迹中的结构进展、不同原子对的注意力富集、距离—时间注意力图及持续接触分析。", "figure.dynamics.context": "图 4 · 分析展示生成过程中结构组织与注意力变化之间的对应关系，不构成对单个模型组件的因果归因。",
    "figure.ablation.title": "消融实验", "figure.ablation.caption": "SALA 状态–上下文框架与结构感知设计的消融实验。", "figure.ablation.alt": "完整 SALA 与七个独立训练变体在构象、体积、晶格形状、堆积和严重原子碰撞上的比较。", "figure.ablation.context": "图 5 · 消融实验基于共同可评估的子集，其中完整模型的堆积 RMSD 为 1.64 Å；基准结果中的 1.70 Å 对应完整的 799 个晶体测试集。",
    "results.eyebrow": "研究结果", "results.title": "生成晶体，检验结构。", "results.description": "以实验晶体为参考，在不同分子组成、结构规模与晶体对称性下评估生成能力。", "results.note": "归档命中率仅按堆积 RMSD 小于 2 Å 统计；联合无碰撞条件的恢复率尚未重新计算。堆积 RMSD 数值为 799 个测试晶体各目标最优误差的均值，评估对象为未经松弛的生成重原子结构。",
    "benchmark.eyebrow": "归档堆积阈值命中统计", "benchmark.summary": "每个目标生成 50 个 SALA 候选，799 个目标中有 575 个的候选堆积 RMSD 小于 2 Å。", "benchmark.link": "查看论文证据", "benchmark.choose": "选择评估指标", "benchmark.recovery": "堆积命中率 ↑", "benchmark.rmsd": "堆积 RMSD ↓", "benchmark.budgets": "每目标候选数：SALA 50 · 全原子方法baseline 50–3,150", "benchmark.unit.recovery": "越高越好 · %", "benchmark.unit.rmsd": "越低越好 · Å", "benchmark.baseline": "全原子方法baseline",
    "paper.authorship": "作者与单位", "paper.affiliation": "北京大学", "paper.title": "通过对称约束的状态–上下文分离生成复杂分子晶体", "paper.title.first": "通过对称约束的状态–上下文分离", "paper.title.second": "生成复杂分子晶体", "paper.abstract.label": "研究概览", "paper.abstract.body": "分子晶体生成需要协同决定分子构象、晶格几何与周期堆积。SALA 采用流匹配，在紧凑且受对称性约束的不对称单元状态中进行演化，同时感知完整的周期环境。这种状态–上下文分离，使模型能够联合生成不同化学类型与晶系下的晶体结构，并在包含四类化学体系、七大晶系的 799 个留出晶体上进行评估。",
    "status.title": "研究状态", "status.lede": "本页展示方法框架、生成结构与论文结果。下列链接将在公开地址确定后开放。",
    "links.note": "模型架构代码已开放，检查点即将开放。", "links.label": "研究链接", "links.paper": "论文地址", "links.code": "代码地址", "links.pending": "链接待添加", "links.checkpoint.pending": "ckpt · 链接待添加", "links.open": "打开链接", "links.checkpoint.open": "模型权重", "links.ready": "已开放", "crystal.download": "下载高清 GIF",
    "texture.word": "对称性",
    "footer.tagline": "以对称性约束状态，以完整环境感知复杂性。", "footer.status": "SALA · 分子晶体生成研究", "footer.top": "返回顶部", "dialog.zoom": "原始尺寸", "dialog.fit": "适应屏幕", "dialog.close": "关闭大图"
  }
};

const crystalData = [
  {id:"859249", type:"single", system:"monoclinic", symmetry:"P2₁/c", width:1600, height:1520},
  {id:"2070206", type:"metal", system:"triclinic", symmetry:"P−1", width:1600, height:1600},
  {id:"726779", type:"single", system:"trigonal", symmetry:"P3₂", width:1600, height:1600},
  {id:"2172461", type:"multi", system:"triclinic", symmetry:"P1", width:1600, height:1600},
  {id:"939712", type:"ionic", system:"trigonal", symmetry:"P3₁", width:1600, height:1600},
  {id:"2128532", type:"metal", system:"orthorhombic", symmetry:"Pca2₁", width:1600, height:1600},
  {id:"1212258", type:"single", system:"tetragonal", symmetry:"P4₁", width:1600, height:1600},
  {id:"1846154", type:"single", system:"hexagonal", symmetry:"P6₁", width:1600, height:1600},
  {id:"1518539", type:"ionic", system:"monoclinic", symmetry:"P2₁", width:1600, height:1600}
];
const figureData = {
  framework:{file:"fig1-framework", number:"01", height:1347},
  performance:{file:"fig2-performance", number:"02", height:1602},
  complexity:{file:"fig3-complexity", number:"03", height:1130},
  dynamics:{file:"fig4-dynamics", number:"04", height:1056},
  ablation:{file:"fig5-ablation", number:"05", height:535}
};
// Each spread's running-head subject, keyed by section id.
const spreadSubjects = {
  overview: "head.subject",
  structures: "subject.structures",
  cases: "subject.cases",
  framework: "subject.framework",
  results: "subject.results",
  figures: "subject.figures",
  status: "subject.status"
};
const TEXTURE_ROWS = 22;

const query = (selector, parent=document) => parent.querySelector(selector);
const all = (selector, parent=document) => [...parent.querySelectorAll(selector)];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const requestedLanguage = new URLSearchParams(location.search).get("lang");
let language = ["en","zh"].includes(requestedLanguage) ? requestedLanguage : "en";
if (!translations[language]) language = "en";
let motionPaused = reducedMotion.matches;
let currentCrystal = 0;
let currentMetric = "recovery";
let currentFigure = "performance";
let currentChemicalType = "all";
let currentCrystalSystem = "all";
let currentSpread = "overview";
let dialogSelection = null;
let previouslyFocused = null;
const t = key => translations[language][key] ?? translations.en[key] ?? key;
const dialog = query("#media-dialog");
const playback = window.createCrystalPlayback({ dialog, paused: () => motionPaused });

function crystalAlt(index) {
  const crystal = crystalData[index];
  return t("crystal.alt.template")
    .replace("{type}", t(`crystal.${crystal.type}.full`))
    .replace("{id}", crystal.id)
    .replace("{symmetry}", crystal.symmetry);
}

function refreshCrystalLabels() {
  all(".structure-card").forEach(card => {
    const index = Number(query("[data-open-crystal]", card).dataset.openCrystal);
    const crystal = crystalData[index];
    query("[data-open-crystal]", card).setAttribute("aria-label", `${t("crystal.enlarge")} ${crystal.id}`);
    query("img", card).alt = crystalAlt(index);
    query("h3", card).textContent = t(`crystal.${crystal.type}`);
    query(".structure-tag", card).textContent = `${t(`system.${crystal.system}`)}*`;
  });
  applyGalleryFilters();
}

function applyGalleryFilters() {
  let count = 0;
  all(".structure-card").forEach(card => {
    const index = Number(query("[data-open-crystal]", card).dataset.openCrystal);
    const crystal = crystalData[index];
    const show = (currentChemicalType === "all" || crystal.type === currentChemicalType) &&
      (currentCrystalSystem === "all" || crystal.system === currentCrystalSystem);
    card.hidden = !show;
    if (show) count += 1;
  });
  all("[data-chemical-type]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.chemicalType === currentChemicalType)));
  query("#crystal-system-filter").value = currentCrystalSystem;
  query("#gallery-count").textContent = t("gallery.count").replace("{shown}", String(count)).replace("{total}", String(crystalData.length));
  query("#gallery-empty").hidden = count > 0;
  measureChrome();
  sweepReveals();
  refreshMotion();
}

function refreshMotion() {
  document.body.classList.toggle("motion-paused", motionPaused);
  all("[data-motion]").forEach(button => {
    button.setAttribute("aria-label", t(motionPaused ? "motion.play" : "motion.pause"));
    button.setAttribute("aria-pressed", String(!motionPaused));
    const label = query("[data-motion-label]", button);
    if(label) label.textContent = t(motionPaused ? "motion.play" : "motion.pause");
  });
  playback.refresh();
}

function selectCrystal(index) {
  currentCrystal = (index + crystalData.length) % crystalData.length;
  const crystal = crystalData[currentCrystal];
  const img = query("#hero-crystal");
  img.src = `assets/crystals/sala-${crystal.id}-tube-preview.webp`;
  playback.setSource(query("#hero-video"), `assets/crystals/sala-${crystal.id}-tube-hero.mp4`);
  img.width = crystal.width;
  img.height = crystal.height;
  img.alt = crystalAlt(currentCrystal);
  query("#hero-counter").textContent = `${String(currentCrystal + 1).padStart(2, "0")} / ${String(crystalData.length).padStart(2, "0")}`;
  query("#hero-type").textContent = t(`crystal.${crystal.type}.full`);
  query("#hero-symmetry").textContent = `${t("crystal.symmetry")} · ${crystal.symmetry}`;
  all("[data-crystal]").forEach(button => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.crystal) === currentCrystal));
    button.setAttribute("aria-label", `${t("crystal.choose")} ${Number(button.dataset.crystal) + 1}`);
  });
  refreshMotion();
}

function selectMetric(metric) {
  currentMetric = metric;
  const isRecovery = metric === "recovery";
  all("[data-metric]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.metric === metric)));
  query("#sala-chart-value").textContent = isRecovery ? "72.0%" : "1.70 Å";
  query("#clari-chart-value").textContent = isRecovery ? "8.1%" : "4.48 Å";
  query("#sala-bar").style.setProperty("--bar", isRecovery ? "72%" : "34%");
  query("#clari-bar").style.setProperty("--bar", isRecovery ? "8.1%" : "89.6%");
  query("#chart-unit").textContent = t(`benchmark.unit.${metric}`);
  query("#chart-mid").textContent = isRecovery ? "50" : "2.5";
  query("#chart-max").textContent = isRecovery ? "100%" : "5 Å";
}

function selectFigure(key) {
  currentFigure = key;
  const figure = figureData[key];
  const img = query("#result-figure-img");
  img.src = `assets/figures/${figure.file}.webp`;
  img.width = 2200;
  img.height = figure.height;
  img.alt = t(`figure.${key}.alt`);
  query("#result-figure-open").dataset.openFigure = key;
  query("#result-figure-number").textContent = `FIG. ${figure.number}`;
  query("#result-figure-caption").textContent = t(`figure.${key}.caption`);
  query("#figure-context").textContent = t(`figure.${key}.context`);
  query("#figure-panel").setAttribute("aria-labelledby", `tab-${key}`);
  all("[data-figure-tab]").forEach(button => {
    const selected = button.dataset.figureTab === key;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
}

function setMenu(open) {
  query(".menu-toggle").setAttribute("aria-expanded", String(open));
  query(".menu-toggle").setAttribute("aria-label", t(open ? "nav.close" : "nav.open"));
  query("#mobile-nav").hidden = !open;
}

function refreshResourceLinks() {
  let ready = 0;
  const configured = window.SALA_LINKS || {};
  all("[data-resource]").forEach(link => {
    const key = link.dataset.resource;
    let url = null;
    try {
      const parsed = new URL(configured[key] || "");
      if (parsed.protocol === "https:") url = parsed.href;
    } catch { /* Empty placeholders are expected until the URLs are supplied. */ }
    const label = query("[data-link-status]", link);
    const slot = query(`[data-status-slot="${key}"]`);
    if (url) {
      ready += 1;
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.removeAttribute("aria-disabled");
      label.textContent = t(key === "checkpoint" ? "links.checkpoint.open" : "links.open");
      if (slot) { slot.textContent = t("links.ready"); slot.classList.add("is-ready"); }
    } else {
      link.removeAttribute("href");
      link.removeAttribute("target");
      link.setAttribute("aria-disabled", "true");
      label.textContent = t(key === "checkpoint" ? "links.checkpoint.pending" : "links.pending");
      if (slot) { slot.textContent = t(key === "checkpoint" ? "links.checkpoint.pending" : "links.pending"); slot.classList.remove("is-ready"); }
    }
  });
  query("#resource-status").hidden = ready === 3;
}

// Repeated-word texture on the closing spread; decorative, rebuilt per language.
function buildWordTexture() {
  const host = query("#word-texture");
  if (!host) return;
  const word = t("texture.word");
  host.textContent = "";
  for (let row = 0; row < TEXTURE_ROWS; row += 1) {
    const span = document.createElement("span");
    // A single sine period across the column, matching the reference page's wave.
    const wave = (Math.sin((row / (TEXTURE_ROWS - 1)) * Math.PI * 2 - Math.PI / 2) + 1) / 2;
    span.style.setProperty("--t", wave.toFixed(3));
    span.textContent = word;
    host.append(span);
  }
}

// The cover title must not break. CSS leaves it wrapping for the no-script
// case; here each line is set to nowrap and shrunk only if it would not fit.
function fitCoverTitle() {
  const heading = query("#cover-heading");
  if (!heading) return;
  heading.classList.add("is-fitted");
  const available = heading.clientWidth;
  if (!available) return;
  all(".title-first, .title-second", heading).forEach(line => {
    line.style.fontSize = "";
    const natural = line.scrollWidth;
    if (!natural || natural <= available) return;
    const base = parseFloat(getComputedStyle(line).fontSize);
    line.style.fontSize = `${Math.max(12, base * (available / natural) * .995).toFixed(2)}px`;
  });
}

function setSpread(id) {
  if (!spreadSubjects[id]) return;
  currentSpread = id;
  const label = query("#spread-label");
  if (label) label.textContent = t(spreadSubjects[id]);
  all(".desktop-nav a, .mobile-nav a, .spread-rail a").forEach(link => {
    if (link.getAttribute("href") === `#${id}`) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
}

// The header sits over the dark cover until the cover scrolls past it.
function refreshHeaderTone() {
  const header = query(".site-header");
  header.classList.toggle("scrolled", window.scrollY > 12);
  header.classList.toggle("on-dark", window.scrollY < coverBottom);
}

// Scroll reveal. A shrinking list is swept on scroll rather than observed, so
// the outcome is deterministic: anything that reaches the viewport is shown,
// and an element can never be left hidden by a missed callback.
let pendingReveals = [];
function sweepReveals() {
  if (!pendingReveals.length) return;
  const limit = window.innerHeight * 0.92;
  pendingReveals = pendingReveals.filter(element => {
    const bounds = element.getBoundingClientRect();
    // A filtered-out card has no box; keep it pending until it is shown again.
    if (bounds.width === 0 && bounds.height === 0) return true;
    if (bounds.top > limit) return true;
    element.classList.add("is-in");
    return false;
  });
}

function armReveals() {
  document.documentElement.dataset.revealArmed = "1";
  measureChrome();
  pendingReveals = all("[data-reveal], .structure-card");
  sweepReveals();
}

// Page geometry is measured on load, resize and after any layout change, so
// the scroll handler itself never reads layout.
let coverBottom = 0;
let scrollRange = 0;
function measureChrome() {
  const root = document.documentElement;
  const cover = query("#overview");
  const header = query(".site-header");
  coverBottom = cover ? cover.offsetTop + cover.offsetHeight - header.offsetHeight : 0;
  scrollRange = Math.max(0, root.scrollHeight - root.clientHeight);
}

function refreshScrollProgress() {
  const ratio = scrollRange > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollRange)) : 0;
  query(".site-header").style.setProperty("--progress", `${(ratio * 100).toFixed(2)}%`);
}

// The cover artwork drifts a little slower than the page as the cover leaves.
function refreshCoverParallax() {
  const art = query(".cover-art");
  const cover = query("#overview");
  if (!art || !cover) return;
  if (reducedMotion.matches) { art.style.transform = ""; return; }
  const progress = Math.min(1, Math.max(0, window.scrollY / Math.max(1, cover.offsetHeight)));
  art.style.transform = `translate3d(0, ${(progress * 58).toFixed(1)}px, 0)`;
}

let scrollFrame = null;
function onScroll() {
  // Cheap, layout-free work runs immediately; the transform write is throttled.
  refreshHeaderTone();
  refreshScrollProgress();
  sweepReveals();
  if (scrollFrame !== null) return;
  scrollFrame = window.requestAnimationFrame(() => {
    scrollFrame = null;
    refreshCoverParallax();
  });
}

function applyLanguage(nextLanguage) {
  language = nextLanguage;
  document.documentElement.lang = language === "zh" ? "zh" : "en";
  document.title = t("meta.title");
  query('meta[name="description"]').content = t("meta.description");
  query('meta[property="og:title"]').content = t("meta.title");
  query('meta[property="og:description"]').content = t("meta.description");
  all("[data-i18n]").forEach(element => {element.textContent = t(element.dataset.i18n);});
  all("[data-i18n-aria]").forEach(element => {element.setAttribute("aria-label", t(element.dataset.i18nAria));});
  all("[data-i18n-alt]").forEach(element => {element.alt = t(element.dataset.i18nAlt);});
  all("[data-lang]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.lang === language)));
  all('.brand,.footer-wordmark').forEach(link => link.setAttribute('aria-label', language === 'zh' ? 'SALA 首页' : 'SALA home'));
  refreshResourceLinks();
  refreshCrystalLabels();
  selectCrystal(currentCrystal);
  selectMetric(currentMetric);
  selectFigure(currentFigure);
  buildWordTexture();
  fitCoverTitle();
  setSpread(currentSpread);
  if(dialog.open) refreshDialog();
  setMenu(query(".menu-toggle").getAttribute("aria-expanded") === "true");
}

function refreshDialog() {
  if(!dialogSelection) return;
  const img = query("#dialog-image");
  const isCrystal = dialogSelection.type === "crystal";
  dialog.classList.toggle("is-crystal", isCrystal);
  query("#dialog-zoom").hidden = false;
  query("#dialog-motion").hidden = !isCrystal;
  query("#dialog-gif").hidden = !isCrystal;
  img.hidden = isCrystal;
  query("#dialog-crystal").hidden = !isCrystal;
  if(isCrystal) {
    const crystal = crystalData[dialogSelection.index];
    query("#dialog-kicker").textContent = `SALA / ${crystal.id}`;
    query("#dialog-title").textContent = `${t(`crystal.${crystal.type}.full`)} · ${crystal.symmetry}`;
    const poster = query("#dialog-crystal-poster");
    poster.src = `assets/crystals/sala-${crystal.id}-tube.webp`;
    poster.alt = crystalAlt(dialogSelection.index);
    poster.width = crystal.width;
    poster.height = crystal.height;
    query("#dialog-crystal").style.setProperty("--crystal-ratio", String(crystal.width / crystal.height));
    playback.setSource(query("#dialog-video"), `assets/crystals/sala-${crystal.id}-tube-detail.mp4`);
    query("#dialog-gif").href = `assets/crystals/sala-${crystal.id}-tube-hd.gif`;
    query("#dialog-gif").download = `SALA-${crystal.id}-tube-1600px.gif`;
    img.alt = crystalAlt(dialogSelection.index);
    img.width = crystal.width;
    img.height = crystal.height;
    query("#dialog-caption").textContent = t("crystal.modal.note");
  } else {
    const {key} = dialogSelection;
    const figure = figureData[key];
    query("#dialog-kicker").textContent = `SALA / FIG. ${figure.number}`;
    query("#dialog-title").textContent = t(`figure.${key}.title`);
    img.src = `assets/figures/${figure.file}.webp`;
    img.alt = t(`figure.${key}.alt`);
    img.width = 2200;
    img.height = figure.height;
    query("#dialog-caption").textContent = t(`figure.${key}.caption`);
  }
  query("#dialog-zoom").textContent = t(query("#dialog-image-wrap").classList.contains("is-actual") ? "dialog.fit" : "dialog.zoom");
  playback.setModal(isCrystal ? query("#dialog-video") : null);
}

function openMedia(selection) {
  previouslyFocused = document.activeElement;
  dialogSelection = selection;
  query("#dialog-image-wrap").classList.remove("is-actual");
  refreshDialog();
  dialog.showModal();
  playback.refresh();
  document.body.style.overflow = "hidden";
  query("#dialog-close").focus();
}

all("[data-resource]").forEach(link => {
  link.addEventListener("click", event => { if (link.getAttribute("aria-disabled") === "true") event.preventDefault(); });
  link.addEventListener("keydown", event => { if (link.getAttribute("aria-disabled") === "true" && ["Enter", " "].includes(event.key)) event.preventDefault(); });
});
all("[data-lang]").forEach(button => button.addEventListener("click", () => applyLanguage(button.dataset.lang)));
all("[data-crystal]").forEach(button => button.addEventListener("click", () => selectCrystal(Number(button.dataset.crystal))));
all("[data-chemical-type]").forEach(button => button.addEventListener("click", () => {
  currentChemicalType = button.dataset.chemicalType;
  applyGalleryFilters();
}));
query("#crystal-system-filter").addEventListener("change", event => {
  currentCrystalSystem = event.target.value;
  applyGalleryFilters();
});
query("#gallery-reset").addEventListener("click", () => {
  currentChemicalType = "all";
  currentCrystalSystem = "all";
  applyGalleryFilters();
});
query("#previous-crystal").addEventListener("click", () => selectCrystal(currentCrystal - 1));
query("#next-crystal").addEventListener("click", () => selectCrystal(currentCrystal + 1));
all("[data-motion]").forEach(button => button.addEventListener("click", () => {
  motionPaused = !motionPaused;
  if (!motionPaused) playback.retry();
  refreshMotion();
}));
all("[data-metric]").forEach(button => button.addEventListener("click", () => selectMetric(button.dataset.metric)));
all("[data-figure-tab]").forEach(button => button.addEventListener("click", () => selectFigure(button.dataset.figureTab)));
query(".figure-tabs").addEventListener("keydown", event => {
  if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
  event.preventDefault();
  const tabs = all("[data-figure-tab]");
  const position = tabs.indexOf(document.activeElement);
  if(position < 0) return;
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length-1 : (position + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus();
  selectFigure(tabs[next].dataset.figureTab);
});
all("[data-open-figure]").forEach(button => button.addEventListener("click", () => openMedia({type:"figure",key:button.dataset.openFigure})));
all("[data-open-crystal]").forEach(button => button.addEventListener("click", () => openMedia({type:"crystal",index:Number(button.dataset.openCrystal)})));
query("#dialog-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => {
  if(event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if(event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
dialog.addEventListener("close", () => {
  document.body.style.overflow = "";
  dialogSelection = null;
  playback.setModal(null);
  query("#dialog-crystal").hidden = true;
  query("#dialog-crystal-poster").removeAttribute("src");
  previouslyFocused?.focus({preventScroll:true});
});
query("#dialog-zoom").addEventListener("click", () => {
  const actual = query("#dialog-image-wrap").classList.toggle("is-actual");
  query("#dialog-zoom").textContent = t(actual ? "dialog.fit" : "dialog.zoom");
});
query(".menu-toggle").addEventListener("click", () => setMenu(query("#mobile-nav").hidden));
all("#mobile-nav a").forEach(link => link.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", event => {if(event.key === "Escape") setMenu(false);});
window.addEventListener("resize", () => {
  if(window.innerWidth > 720) setMenu(false);
  fitCoverTitle();
  measureChrome();
  refreshHeaderTone();
  refreshScrollProgress();
  sweepReveals();
}, {passive:true});
window.addEventListener("scroll", onScroll, {passive:true});
reducedMotion.addEventListener("change", event => {motionPaused = event.matches;refreshMotion();refreshCoverParallax();});

if("IntersectionObserver" in window) {
  const spreadObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => { if(entry.isIntersecting) setSpread(entry.target.id); });
  }, {rootMargin:"-20% 0px -62% 0px", threshold:0});
  all("#main > [id]").forEach(spread => spreadObserver.observe(spread));
}
applyLanguage(language);
measureChrome();
armReveals();
refreshHeaderTone();
refreshScrollProgress();
refreshCoverParallax();
