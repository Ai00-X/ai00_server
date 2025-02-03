import tkinter as tk
from tkinter import ttk, messagebox
import json
from pydantic import create_model
from typing import List, Optional
from formatron.schemas.pydantic import ClassSchema
from formatron.formatter import FormatterBuilder
from kbnf.engine import Vocabulary
import random

# Step 1: Import the random module
class JsonSchemaDesigner:
    def __init__(self, root):
        self.root = root
        self.root.title("make KBNF form JSON Schema  -- made by AI00")
        self.root.geometry("800x600")
        self.root.configure(padx=10, pady=10)

        # Step 2: Store the base part of the title
        self.base_title = "make KBNF form JSON Schema  -- made by "

        # Step 3: Create a list of the variations
        self.title_variations = [ "AI0-", "AI--", "AI-0"]

        # Step 4: Define a function to update the title
        self.oks = False
        def animate_title():
            # Select a random variation
             
            if self.oks == False:
                new_title = self.base_title + random.choice(self.title_variations)
                
            else:
                new_title = self.base_title + "AI00"

            self.oks = not self.oks
            self.root.title(new_title)
            # Schedule the next title change after 500 milliseconds
            self.root.after(1000, animate_title)

        # Step 5: Start the animation
        animate_title()
        # 主Frame，使用pack布局，填满整个窗口
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # PanedWindow用于垂直分割TreeView和文本框，放在主Frame中
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # TreeView面板
        self.tree_container = ttk.Frame(self.paned_window)
        self.paned_window.add(self.tree_container, weight=1)

        # 嵌套Frame，用于TreeView和按钮
        self.tree_subframe = ttk.Frame(self.tree_container)
        self.tree_subframe.pack(fill=tk.BOTH, expand=True)

        # TreeView
        self.tree = ttk.Treeview(self.tree_subframe, columns=("Type"), show="tree headings")
        self.tree.heading("#0", text="Key")
        self.tree.heading("Type", text="Type")
        self.tree.grid(row=0, column=0, sticky="nsew")

        # TreeView垂直滚动条
        self.tree_scrollbar = ttk.Scrollbar(self.tree_subframe, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=self.tree_scrollbar.set)

        # 按钮：Generate KBNF
        self.generate_button = tk.Button(self.tree_subframe, text="Generate KBNF", command=self.generate_json)
        self.generate_button.grid(row=1, column=0, sticky="ew")

        # 设置tree_subframe的列和行权重
        self.tree_subframe.grid_columnconfigure(0, weight=1)
        self.tree_subframe.grid_rowconfigure(0, weight=1)

        # 文本框面板
        self.text_container = ttk.Frame(self.paned_window)
        self.paned_window.add(self.text_container, weight=1)

        # 嵌套Frame，用于文本框和按钮
        self.text_subframe = ttk.Frame(self.text_container)
        self.text_subframe.pack(fill=tk.BOTH, expand=True)

        # 文本框
        self.kbnf_text = tk.Text(self.text_subframe, height=10, wrap=tk.WORD)
        self.kbnf_text.grid(row=0, column=0, sticky="nsew")

        # 文本框垂直滚动条
        self.text_scrollbar = ttk.Scrollbar(self.text_subframe, orient=tk.VERTICAL, command=self.kbnf_text.yview)
        self.text_scrollbar.grid(row=0, column=1, sticky="ns")
        self.kbnf_text.configure(yscrollcommand=self.text_scrollbar.set)

        # 按钮：Copy to Clipboard
        self.copy_button = tk.Button(self.text_subframe, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.grid(row=1, column=0, sticky="ew")

        # 设置text_subframe的列和行权重
        self.text_subframe.grid_columnconfigure(0, weight=1)
        self.text_subframe.grid_rowconfigure(0, weight=1)

        # 右键菜单
        self.item_context_menu = tk.Menu(self.root, tearoff=0)
        self.item_context_menu.add_command(label="Modify", command=self.edit_key)
        self.item_context_menu.add_command(label="Delete", command=self.delete_key)
        self.item_context_menu.add_command(label="Add Child", command=lambda: self.add_key(self.tree.selection()[0]))

        self.root_context_menu = tk.Menu(self.root, tearoff=0)
        self.root_context_menu.add_command(label="Add Key", command=lambda: self.add_key(""))

        self.tree.bind("<Button-3>", self.show_context_menu)

        self.edit_window = None

    def show_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        selected_items = self.tree.selection()

        if item:
            if len(selected_items) > 1:
                menu = tk.Menu(self.root, tearoff=0)
                menu.add_command(label="Delete", command=self.delete_key)
                menu.post(event.x_root, event.y_root)
            else:
                self.item_context_menu.post(event.x_root, event.y_root)
        else:
            self.root_context_menu.post(event.x_root, event.y_root)

    def add_key(self, parent=""):
        self._edit_key(None, parent)

    def edit_key(self):
        selected_item = self.tree.selection()
        if selected_item:
            self._edit_key(selected_item[0])
        else:
            messagebox.showwarning("No Selection", "Please select a key to edit.")

    def _edit_key(self, item, parent=""):
        if self.edit_window is not None:
            return

        def save():
            key = key_entry.get()
            data_type = type_combobox.get()
            if key and data_type:
                if item:
                    self.tree.item(item, text=key, values=(data_type,))
                    self.tree.item(item, tags=(data_type,))
                else:
                    new_item = self.tree.insert(parent, "end", text=key, values=(data_type,), tags=(data_type,))
            on_close()

        def on_close():
            self.edit_window = None
            edit_window.destroy()

        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Key" if item else "Add Key")
        edit_window.resizable(False, False)
        x = self.root.winfo_x() + 50
        y = self.root.winfo_y() + 50
        edit_window.geometry(f"+{x}+{y}")
        edit_window.configure(padx=10, pady=10)

        tk.Label(edit_window, text="Key:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        key_entry = tk.Entry(edit_window, width=30)
        key_entry.grid(row=0, column=1, sticky="ew", pady=(0, 5))
        key_entry.focus_set()

        tk.Label(edit_window, text="Type:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        type_combobox = ttk.Combobox(edit_window, values=["str", "int", "bool", "obj", "number", "null"], width=27)
        type_combobox.grid(row=1, column=1, sticky="ew", pady=(0, 5))
        type_combobox.current(0)

        if item:
            key_entry.insert(0, self.tree.item(item, "text"))
            type_combobox.set(self.tree.item(item, "values")[0])

        button_frame = tk.Frame(edit_window)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        tk.Button(button_frame, text="Save", command=save).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_close).pack(side=tk.RIGHT)

        edit_window.grab_set()
        self.edit_window = edit_window
        edit_window.protocol("WM_DELETE_WINDOW", on_close)

    def delete_key(self):
        selected_items = self.tree.selection()
        if selected_items:
            for item in selected_items:
                self.tree.delete(item)
        else:
            messagebox.showwarning("No Selection", "Please select a key to delete.")

    def generate_json(self):
        def build_schema(item):
            schema = {}
            for child in self.tree.get_children(item):
                key = self.tree.item(child, "text")
                data_type = self.tree.item(child, "values")[0]
                if data_type == "obj":
                    schema[key] = build_schema(child)
                else:
                    schema[key] = data_type
            return schema

        schema = build_schema("")
        json_output = {"json": schema}

        DynamicClass = self.generate_class_schema(json_output)

        vocabulary = Vocabulary({}, {})
        decode = lambda x: x
        f = FormatterBuilder()
        f.append_line(f"{f.json(DynamicClass, capture_name='json')}")
        ss = f.build(vocabulary=vocabulary, decode=decode).grammar_str

        self.kbnf_text.delete(1.0, tk.END)
        self.kbnf_text.insert(tk.END, ss)

    def generate_class_schema(self, json_output):
        def map_type(data_type):
            if data_type == "str":
                return str
            elif data_type == "int":
                return int
            elif data_type == "bool":
                return bool
            elif data_type == "obj":
                return dict
            elif data_type == "number":
                return float
            elif data_type == "null":
                return type(None)
            else:
                raise ValueError(f"Unknown type: {data_type}")

        def build_class_schema(schema, class_name="DynamicClass"):
            fields = {}
            for key, value in schema.items():
                if isinstance(value, dict):
                    fields[key] = (build_class_schema(value, key.title()), ...)
                else:
                    field_type = map_type(value)
                    fields[key] = (field_type, ...)
            return create_model(class_name, **fields, __base__=ClassSchema)

        schema = json_output["json"]
        return build_class_schema(schema, "Goods")

    def copy_to_clipboard(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.kbnf_text.get(1.0, tk.END))
        messagebox.showinfo("Copied", "kbnf grammar copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = JsonSchemaDesigner(root)
    root.mainloop()