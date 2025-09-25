"""
MuJoCo场景编辑器主入口

初始化应用程序并连接各个组件。
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QMessageBox, QFileDialog, QAction
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

# 导入视图模型
from .viewmodel.scene_viewmodel import SceneViewModel
from .viewmodel.property_viewmodel import PropertyViewModel
from .viewmodel.hierarchy_viewmodel import HierarchyViewModel
from .viewmodel.control_viewmodel import ControlViewModel
from .model.xml_parser import XMLParser

# 导入视图组件
from .view.opengl_view import OpenGLView
from .view.property_panel import PropertyPanel
from .view.hierarchy_tree import HierarchyTree
from .view.control_panel import ControlPanel

class MainWindow(QMainWindow):
    """
    主窗口类
    
    管理应用程序的所有UI组件和视图模型
    """
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("MuJoCo场景编辑器")
        self.resize(1200, 800)
        
        # 创建视图模型
        self.scene_viewmodel = SceneViewModel()
        self.property_viewmodel = PropertyViewModel(self.scene_viewmodel)
        self.hierarchy_viewmodel = HierarchyViewModel(self.scene_viewmodel)
        self.control_viewmodel = ControlViewModel(self.scene_viewmodel)
        
        # 创建视图组件
        self.opengl_view = OpenGLView(self.scene_viewmodel)
        self.property_panel = PropertyPanel(self.property_viewmodel)
        self.hierarchy_tree = HierarchyTree(self.hierarchy_viewmodel)
        self.control_panel = ControlPanel(self.control_viewmodel)
        #LZQ：0904
        # 初始化取 VM 里的值（取不到就用 1.0）
        try:
            self.control_panel.gizmoGlobalSpin.setValue(self.scene_viewmodel.global_gizmo_size_world)
        except Exception:
            self.control_panel.gizmoGlobalSpin.setValue(1.0)

        # 改动即生效
        self.control_panel.gizmoGlobalSpin.valueChanged.connect(
            lambda v: self.scene_viewmodel.set_global_gizmo_size_world(v)
        )
        self.scene_viewmodel.gizmoSizeChanged.connect(self._on_gizmo_size_changed)
        
        # 设置中央窗口部件
        self.setCentralWidget(self.opengl_view)
        
        # 添加停靠窗口
        self._setup_dock_widgets()
        
        # 创建菜单栏
        self._create_menus()
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        
        # 记录当前打开的文件
        self.current_file = None
        self.loaded_xml_files = []

        # ply修改建立连接
        # 控制面板发出的GS编辑、选择变化都要同步给OpenGL视图
        # GS 编辑：UI -> OpenGL 视图，参数包含平移/旋转/缩放
        self.control_panel.applyGsEditRequested.connect(self.opengl_view.apply_gsply_transform)
        self.control_panel.gsPlySelectionChanged.connect(self.opengl_view.set_active_gs_background)
        # 初始化下拉框条目，与当前已加载的GS背景保持一致
        self.control_panel.set_gs_ply_entries(self.opengl_view.get_gs_background_entries(), emit_change=False)
        # 监听SceneViewModel，确保任何层的变动都能刷新界面
        self.scene_viewmodel.gsBackgroundsChanged.connect(self._on_gs_backgrounds_changed)
        self.scene_viewmodel.control_viewmodel = self.control_viewmodel

    
    def _on_gizmo_size_changed(self, value: float):
        self.control_panel.gizmoGlobalSpin.blockSignals(True)
        self.control_panel.gizmoGlobalSpin.setValue(value)
        self.control_panel.gizmoGlobalSpin.blockSignals(False)

    def _on_gs_backgrounds_changed(self, entries, active_key):
        # 当场景层面更新高斯背景列表时，让渲染视图及时重载PLY，[(key, path), ...]
        self.opengl_view.update_gs_backgrounds_from_scene(entries, active_key)
        combo_entries = [(item.get('key'), item.get('path')) for item in entries]
        # 将最新的高斯背景同步到控制面板的下拉列表
        self.control_panel.set_gs_ply_entries(combo_entries, emit_change=False)
        if active_key:
            # 维持当前激活的背景选择，不触发额外信号
            self.control_panel.select_gs_key(active_key, emit=False)
        else:
            # 无激活项时清空选中状态，同时避免产生多余的切换事件
            self.control_panel.gs_file_combo.blockSignals(True)
            self.control_panel.gs_file_combo.setCurrentIndex(-1)
            self.control_panel.gs_file_combo.blockSignals(False)

    
    def _setup_dock_widgets(self):
        """设置停靠窗口"""
        # 层级树面板（左侧）
        hierarchy_dock = QDockWidget("层级结构", self)
        hierarchy_dock.setWidget(self.hierarchy_tree)
        hierarchy_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, hierarchy_dock)
        
        # 控制面板（左侧）
        control_dock = QDockWidget("控制面板", self)
        control_dock.setWidget(self.control_panel)
        control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, control_dock)
        
        # 属性面板（右侧）
        property_dock = QDockWidget("属性", self)
        property_dock.setWidget(self.property_panel)
        property_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, property_dock)
    
    def _create_menus(self):
        """创建菜单"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件(&F)")
        
        # 新建场景
        new_action = QAction("新建(&N)", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._new_scene)
        file_menu.addAction(new_action)
        
        # 打开
        open_action = QAction("打开(&O)...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        # 打开
        open_action_ply = QAction("打开(&ply)...", self)
        # open_action_ply.setShortcut(QKeySequence.Open)
        open_action_ply.triggered.connect(self._open_file_ply)
        file_menu.addAction(open_action_ply)

        # 1) 菜单里添加一个“打开 OBJ/STL”
        open_mesh_action = QAction("打开(&OBJ/STL)...", self)
        # open_mesh_action.setShortcut("Ctrl+Shift+O")
        open_mesh_action.triggered.connect(self._open_file_mesh)
        file_menu.addAction(open_mesh_action)
       
        # 分隔线
        file_menu.addSeparator()
        
        # 保存
        save_action = QAction("保存(&S)", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)
        
        # 另存为
        save_as_action = QAction("另存为(&A)...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._save_file_as)
        file_menu.addAction(save_as_action)
        
        # 分隔线
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出(&Q)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = self.menuBar().addMenu("编辑(&E)")
        
        # 撤销/重做
        self.undo_action = QAction("撤销(&U)", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self._undo)
        edit_menu.addAction(self.undo_action)
        self.undo_action.setEnabled(False)  # 初始状态禁用
        
        self.redo_action = QAction("重做(&R)", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self._redo)
        edit_menu.addAction(self.redo_action)
        self.redo_action.setEnabled(False)  # 初始状态禁用
        
        # 连接撤销/重做状态变化信号
        self.control_viewmodel.undoStateChanged.connect(self._update_undo_state)
        self.control_viewmodel.redoStateChanged.connect(self._update_redo_state)
        
        # 分隔线
        edit_menu.addSeparator()
        
        # 复制
        copy_action = QAction("复制(&C)", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self._copy)
        edit_menu.addAction(copy_action)
        
        # 粘贴
        paste_action = QAction("粘贴(&P)", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self._paste)
        edit_menu.addAction(paste_action)
        
        # 删除
        delete_action = QAction("删除(&D)", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self._delete)
        edit_menu.addAction(delete_action)
        
        # 视图菜单
        view_menu = self.menuBar().addMenu("视图(&V)")
        
        # 重置视图
        reset_view_action = QAction("重置视图", self)
        reset_view_action.triggered.connect(self._reset_all_views)
        view_menu.addAction(reset_view_action)
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助(&H)")
        
        # 关于
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _new_scene(self):
        """创建新场景"""
        # 提示保存当前场景
        if len(self.scene_viewmodel.geometries) > 0:
            reply = QMessageBox.question(
                self, "创建新场景", 
                "创建新场景将丢失当前场景的所有更改。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # 清空当前场景
        self.scene_viewmodel.clear_scene()
        # 重置当前文件
        self.current_file = None
        self.loaded_xml_files = []
        # 更新窗口标题
        self.setWindowTitle("MuJoCo场景编辑器")
        self.statusBar().showMessage("已创建新场景")
    
    def _open_file(self):
        """打开文件"""
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "打开场景", "", "XML文件 (*.xml);;所有文件 (*)"
        )

        if not filenames:
            return

        # 如果场景已有内容，提示是否替换
        if len(self.scene_viewmodel.geometries) > 0:
            reply = QMessageBox.question(
                self, "打开场景",
                "打开新的XML会清空当前场景。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.scene_viewmodel.clear_scene()
        self.opengl_view.clear_loaded_meshes()

        loaded_paths = []
        failed_paths = []
        for index, path in enumerate(filenames):
            ok = self.scene_viewmodel.load_scene(path, append=(index > 0))
            if ok:
                loaded_paths.append(os.path.abspath(path))
            else:
                failed_paths.append(path)

        if loaded_paths:
            self.loaded_xml_files = loaded_paths
            if len(loaded_paths) == 1:
                self.current_file = loaded_paths[0]
                title_suffix = os.path.basename(loaded_paths[0])
            else:
                self.current_file = None
                title_suffix = f"{len(loaded_paths)} 个文件"

            self.setWindowTitle(f"MuJoCo场景编辑器 - {title_suffix}")
            self.statusBar().showMessage(f"已加载 {len(loaded_paths)} 个XML")
        else:
            self.statusBar().showMessage("未加载任何XML")

        if failed_paths:
            QMessageBox.warning(self, "加载错误", "无法加载以下XML：\n" + "\n".join(failed_paths))
    
    def _open_file_ply(self):
        """打开 ply 文件，可一次选择多个"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择 PLY 文件", "", "PLY 文件 (*.ply)"
        )
        if not paths:
            return

        self.opengl_view.set_gs_backgrounds(paths, reset_history=True)
        self.statusBar().showMessage(f"已加载 {len(paths)} 个 PLY 文件")

    def _open_file_mesh(self):
        """打开 OBJ / STL 文件并让 OpenGL 视图显示"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 3D 模型文件",
            "",
            "3D模型 (*.obj *.stl);;OBJ 文件 (*.obj);;STL 文件 (*.stl)"
        )
        # 仅加载单个网格作为临时可视化对象，不写入当前 XML 场景
        if not filename:
            return

        ok = self.opengl_view.load_mesh(filename)
        if ok:
            self.statusBar().showMessage(f"已加载模型: {os.path.basename(filename)}")
        else:
            QMessageBox.warning(self, "加载失败", f"无法加载: {os.path.basename(filename)}")

    def _save_file(self):
        """保存文件"""
        sources = self.scene_viewmodel.get_loaded_sources()
        sources = [os.path.abspath(p) for p in sources]

        if len(sources) <= 1:
            target = self.current_file or (sources[0] if sources else None)
            if target:
                if self.scene_viewmodel.save_scene(target):
                    self.statusBar().showMessage(f"已保存场景: {os.path.basename(target)}")
                else:
                    QMessageBox.warning(self, "保存错误", "无法保存场景文件。")
            else:
                # 如果没有当前文件，则调用另存为
                self._save_file_as()
            return

        if self.scene_viewmodel.has_unsourced_geometry():
            QMessageBox.information(
                self,
                "提示",
                "存在未关联来源的几何体，请使用“另存为”导出它们。"
            )

        failures = self.scene_viewmodel.save_loaded_sources()
        if failures:
            QMessageBox.warning(self, "保存错误", "以下XML保存失败：\n" + "\n".join(failures))
        else:
            self.statusBar().showMessage(f"已保存 {len(sources)} 个XML")
    
    def _save_file_as(self):
        """另存为"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存场景", "", "XML文件 (*.xml);;所有文件 (*)"
        )
        
        if filename:
            if not filename.lower().endswith(('.xml')):
                filename += '.xml'
                
            export_unsourced_only = self.scene_viewmodel.has_unsourced_geometry()
            if self.scene_viewmodel.save_scene(filename, include_unsourced_only=export_unsourced_only):
                # 更新当前文件
                self.current_file = filename
                abs_path = os.path.abspath(filename)
                if abs_path not in self.loaded_xml_files:
                    self.loaded_xml_files.append(abs_path)
                # 更新窗口标题
                self.setWindowTitle(f"MuJoCo场景编辑器 - {os.path.basename(filename)}")
                self.statusBar().showMessage(f"已保存场景: {os.path.basename(filename)}")

                if export_unsourced_only:
                    failures = self.scene_viewmodel.save_loaded_sources()
                    if failures:
                        QMessageBox.warning(self, "保存错误", "以下XML保存失败：\n" + "\n".join(failures))
                    else:
                        self.statusBar().showMessage(f"已保存场景: {os.path.basename(filename)}，并同步更新所有打开的XML")
            else:
                QMessageBox.warning(self, "保存错误", "无法保存场景文件。")
    
    def _copy(self):
        """复制当前选中的几何体"""
        selected = self.scene_viewmodel.selected_geometry
        if selected:
            if self.hierarchy_viewmodel.copy_geometry(selected):
                self.statusBar().showMessage(f"已复制: {selected.name}")
    
    def _paste(self):
        """粘贴几何体"""
        result = self.hierarchy_viewmodel.paste_geometry()
        if result:
            self.statusBar().showMessage(f"已粘贴: {result.name}")
    
    def _delete(self):
        """删除当前选中的几何体"""
        selected = self.scene_viewmodel.selected_geometry
        if selected:
            name = selected.name
            self.hierarchy_viewmodel.remove_geometry(selected)
            self.statusBar().showMessage(f"已删除: {name}")
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, 
            "关于MuJoCo场景编辑器", 
            "MuJoCo场景编辑器 v0.1.0\n\n"
            "一个用于创建和编辑MuJoCo场景的图形界面工具。"
        )
    
    def _reset_all_views(self):
        """重置所有视图"""
        # 重置3D视图
        self.opengl_view.reset_camera()
        
        # 重置并确保属性面板可见
        self.property_viewmodel.reset_properties()
        
        # 需要检查的面板及其重建方法
        dock_panels = {
            "属性": self._recreate_property_panel,
            "控制面板": self._recreate_control_panel,
            "层级结构": self._recreate_hierarchy_panel,
        }
        # 记录哪些面板已找到
        found_panels = {key: False for key in dock_panels}
        
        for dock in self.findChildren(QDockWidget):
            title = dock.windowTitle()
            if title in dock_panels:
                found_panels[title] = True
                if not dock.isVisible():
                    dock.setVisible(True)
                    dock.raise_()
        
        # 没有找到的面板，重新创建
        for panel, recreate_func in dock_panels.items():
            if not found_panels[panel]:
                recreate_func()
        
        # 通知状态栏
        self.statusBar().showMessage("已重置所有视图")
    
    def _recreate_property_panel(self):
        """重新创建属性面板"""
        # 创建新的属性面板
        property_dock = QDockWidget("属性", self)
        property_dock.setWidget(self.property_panel)
        property_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, property_dock)
    
    def _recreate_control_panel(self):
        """重新创建控制面板"""
        control_dock = QDockWidget("控制面板", self)
        control_dock.setWidget(self.control_panel)
        control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, control_dock)
    
    def _recreate_hierarchy_panel(self):
        """重新创建层级结构面板"""
        hierarchy_dock = QDockWidget("层级结构", self)
        hierarchy_dock.setWidget(self.hierarchy_tree)
        hierarchy_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, hierarchy_dock)
    
    def _undo(self):
        """执行撤销操作"""
        if self.control_viewmodel.can_undo():
            success = self.control_viewmodel.undo()
            if success:
                self.statusBar().showMessage("已撤销操作")
            else:
                self.statusBar().showMessage("撤销操作失败")

    def _redo(self):
        """执行重做操作"""
        if self.control_viewmodel.can_redo():
            success = self.control_viewmodel.redo()
            if success:
                self.statusBar().showMessage("已重做操作")
            else:
                self.statusBar().showMessage("重做操作失败")

    def _update_undo_state(self, can_undo):
        """更新撤销按钮状态"""
        self.undo_action.setEnabled(can_undo)

    def _update_redo_state(self, can_redo):
        """更新重做按钮状态"""
        self.redo_action.setEnabled(can_redo)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 提示保存
        if len(self.scene_viewmodel.geometries) > 0:
            reply = QMessageBox.question(
                self, "退出程序", 
                "是否保存当前场景？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                self._save_file()
                # 保存完成后清理历史记录
                self.control_viewmodel.clear_history()
                if hasattr(self.scene_viewmodel, 'clear_gs_backups'):
                    self.scene_viewmodel.clear_gs_backups()
                event.accept()
            elif reply == QMessageBox.Discard:
                # 不保存但仍需清理历史记录
                self.control_viewmodel.clear_history()
                if hasattr(self.scene_viewmodel, 'clear_gs_backups'):
                    self.scene_viewmodel.clear_gs_backups()
                event.accept()
            else:
                event.ignore()
        else:
            # 没有几何体也需要清理历史记录
            self.control_viewmodel.clear_history()
            if hasattr(self.scene_viewmodel, 'clear_gs_backups'):
                self.scene_viewmodel.clear_gs_backups()
            event.accept()

def main():
    """应用程序入口点"""
    # 设置应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("MuJoCo场景编辑器")
    
    # 启用OpenGL
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
