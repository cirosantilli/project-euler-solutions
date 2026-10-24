import ProjectEulerSolutions.Lib.PrimePairLarge.Graph014
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row7643_eq_data : row7643 = Data.row7643 := rfl
def node7643 : Graph := .node 7643 row7643 node7639 Graph.empty
set_option maxRecDepth 4096 in
theorem node7643_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7643 :=
  Budget.node_covers _ _ 7643 row7643 node7639 Graph.empty
    node7639_covers empty_covers (by decide) (by decide) row7643_covers
theorem node7643_eq_data : node7643 = Data.node7643 := by
  have branches : (node7639, Graph.empty) = (Data.node7639, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7639_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7643 row children.1 children.2) row7643_eq_data branches
set_option maxRecDepth 4096 in
theorem row7621_eq_data : row7621 = Data.row7621 := rfl
def node7621 : Graph := .node 7621 row7621 node7607 node7643
set_option maxRecDepth 4096 in
theorem node7621_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7621 :=
  Budget.node_covers _ _ 7621 row7621 node7607 node7643
    node7607_covers node7643_covers (by decide) (by decide) row7621_covers
theorem node7621_eq_data : node7621 = Data.node7621 := by
  have branches : (node7607, node7643) = (Data.node7607, Data.node7643) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7607_eq_data node7643_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7621 row children.1 children.2) row7621_eq_data branches
set_option maxRecDepth 4096 in
theorem row7591_eq_data : row7591 = Data.row7591 := rfl
def node7591 : Graph := .node 7591 row7591 node7577 node7621
set_option maxRecDepth 4096 in
theorem node7591_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7591 :=
  Budget.node_covers _ _ 7591 row7591 node7577 node7621
    node7577_covers node7621_covers (by decide) (by decide) row7591_covers
theorem node7591_eq_data : node7591 = Data.node7591 := by
  have branches : (node7577, node7621) = (Data.node7577, Data.node7621) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7577_eq_data node7621_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7591 row children.1 children.2) row7591_eq_data branches
set_option maxRecDepth 4096 in
theorem row7559_eq_data : row7559 = Data.row7559 := rfl
def node7559 : Graph := .node 7559 row7559 node7523 node7591
set_option maxRecDepth 4096 in
theorem node7559_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7559 :=
  Budget.node_covers _ _ 7559 row7559 node7523 node7591
    node7523_covers node7591_covers (by decide) (by decide) row7559_covers
theorem node7559_eq_data : node7559 = Data.node7559 := by
  have branches : (node7523, node7591) = (Data.node7523, Data.node7591) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7523_eq_data node7591_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7559 row children.1 children.2) row7559_eq_data branches
set_option maxRecDepth 4096 in
theorem row7481_eq_data : row7481 = Data.row7481 := rfl
def node7481 : Graph := .node 7481 row7481 node7333 node7559
set_option maxRecDepth 4096 in
theorem node7481_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7481 :=
  Budget.node_covers _ _ 7481 row7481 node7333 node7559
    node7333_covers node7559_covers (by decide) (by decide) row7481_covers
theorem node7481_eq_data : node7481 = Data.node7481 := by
  have branches : (node7333, node7559) = (Data.node7333, Data.node7559) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7333_eq_data node7559_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7481 row children.1 children.2) row7481_eq_data branches
set_option maxRecDepth 4096 in
theorem row7219_eq_data : row7219 = Data.row7219 := rfl
def node7219 : Graph := .node 7219 row7219 node6991 node7481
set_option maxRecDepth 4096 in
theorem node7219_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7219 :=
  Budget.node_covers _ _ 7219 row7219 node6991 node7481
    node6991_covers node7481_covers (by decide) (by decide) row7219_covers
theorem node7219_eq_data : node7219 = Data.node7219 := by
  have branches : (node6991, node7481) = (Data.node6991, Data.node7481) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6991_eq_data node7481_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7219 row children.1 children.2) row7219_eq_data branches
set_option maxRecDepth 4096 in
theorem row6791_eq_data : row6791 = Data.row6791 := rfl
def node6791 : Graph := .node 6791 row6791 node6343 node7219
set_option maxRecDepth 4096 in
theorem node6791_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6791 :=
  Budget.node_covers _ _ 6791 row6791 node6343 node7219
    node6343_covers node7219_covers (by decide) (by decide) row6791_covers
theorem node6791_eq_data : node6791 = Data.node6791 := by
  have branches : (node6343, node7219) = (Data.node6343, Data.node7219) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6343_eq_data node7219_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6791 row children.1 children.2) row6791_eq_data branches
set_option maxRecDepth 4096 in
theorem row7669_eq_data : row7669 = Data.row7669 := rfl
def node7669 : Graph := .node 7669 row7669 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7669_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7669 :=
  Budget.node_covers _ _ 7669 row7669 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7669_covers
theorem node7669_eq_data : node7669 = Data.node7669 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7669 row children.1 children.2) row7669_eq_data branches
set_option maxRecDepth 4096 in
theorem row7681_eq_data : row7681 = Data.row7681 := rfl
def node7681 : Graph := .node 7681 row7681 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7681_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7681 :=
  Budget.node_covers _ _ 7681 row7681 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7681_covers
theorem node7681_eq_data : node7681 = Data.node7681 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7681 row children.1 children.2) row7681_eq_data branches
set_option maxRecDepth 4096 in
theorem row7673_eq_data : row7673 = Data.row7673 := rfl
def node7673 : Graph := .node 7673 row7673 node7669 node7681
set_option maxRecDepth 4096 in
theorem node7673_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7673 :=
  Budget.node_covers _ _ 7673 row7673 node7669 node7681
    node7669_covers node7681_covers (by decide) (by decide) row7673_covers
theorem node7673_eq_data : node7673 = Data.node7673 := by
  have branches : (node7669, node7681) = (Data.node7669, Data.node7681) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7669_eq_data node7681_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7673 row children.1 children.2) row7673_eq_data branches
set_option maxRecDepth 4096 in
theorem row7691_eq_data : row7691 = Data.row7691 := rfl
def node7691 : Graph := .node 7691 row7691 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7691_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7691 :=
  Budget.node_covers _ _ 7691 row7691 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7691_covers
theorem node7691_eq_data : node7691 = Data.node7691 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7691 row children.1 children.2) row7691_eq_data branches
set_option maxRecDepth 4096 in
theorem row7699_eq_data : row7699 = Data.row7699 := rfl
def node7699 : Graph := .node 7699 row7699 node7691 Graph.empty
set_option maxRecDepth 4096 in
theorem node7699_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7699 :=
  Budget.node_covers _ _ 7699 row7699 node7691 Graph.empty
    node7691_covers empty_covers (by decide) (by decide) row7699_covers
theorem node7699_eq_data : node7699 = Data.node7699 := by
  have branches : (node7691, Graph.empty) = (Data.node7691, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7691_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7699 row children.1 children.2) row7699_eq_data branches
set_option maxRecDepth 4096 in
theorem row7687_eq_data : row7687 = Data.row7687 := rfl
def node7687 : Graph := .node 7687 row7687 node7673 node7699
set_option maxRecDepth 4096 in
theorem node7687_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7687 :=
  Budget.node_covers _ _ 7687 row7687 node7673 node7699
    node7673_covers node7699_covers (by decide) (by decide) row7687_covers
theorem node7687_eq_data : node7687 = Data.node7687 := by
  have branches : (node7673, node7699) = (Data.node7673, Data.node7699) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7673_eq_data node7699_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7687 row children.1 children.2) row7687_eq_data branches
set_option maxRecDepth 4096 in
theorem row7717_eq_data : row7717 = Data.row7717 := rfl
def node7717 : Graph := .node 7717 row7717 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7717_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7717 :=
  Budget.node_covers _ _ 7717 row7717 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7717_covers
theorem node7717_eq_data : node7717 = Data.node7717 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7717 row children.1 children.2) row7717_eq_data branches
set_option maxRecDepth 4096 in
theorem row7723_eq_data : row7723 = Data.row7723 := rfl
def node7723 : Graph := .node 7723 row7723 node7717 Graph.empty
set_option maxRecDepth 4096 in
theorem node7723_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7723 :=
  Budget.node_covers _ _ 7723 row7723 node7717 Graph.empty
    node7717_covers empty_covers (by decide) (by decide) row7723_covers
theorem node7723_eq_data : node7723 = Data.node7723 := by
  have branches : (node7717, Graph.empty) = (Data.node7717, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7717_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7723 row children.1 children.2) row7723_eq_data branches
set_option maxRecDepth 4096 in
theorem row7741_eq_data : row7741 = Data.row7741 := rfl
def node7741 : Graph := .node 7741 row7741 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7741_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7741 :=
  Budget.node_covers _ _ 7741 row7741 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7741_covers
theorem node7741_eq_data : node7741 = Data.node7741 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7741 row children.1 children.2) row7741_eq_data branches
set_option maxRecDepth 4096 in
theorem row7753_eq_data : row7753 = Data.row7753 := rfl
def node7753 : Graph := .node 7753 row7753 node7741 Graph.empty
set_option maxRecDepth 4096 in
theorem node7753_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7753 :=
  Budget.node_covers _ _ 7753 row7753 node7741 Graph.empty
    node7741_covers empty_covers (by decide) (by decide) row7753_covers
theorem node7753_eq_data : node7753 = Data.node7753 := by
  have branches : (node7741, Graph.empty) = (Data.node7741, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7741_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7753 row children.1 children.2) row7753_eq_data branches
set_option maxRecDepth 4096 in
theorem row7727_eq_data : row7727 = Data.row7727 := rfl
def node7727 : Graph := .node 7727 row7727 node7723 node7753
set_option maxRecDepth 4096 in
theorem node7727_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7727 :=
  Budget.node_covers _ _ 7727 row7727 node7723 node7753
    node7723_covers node7753_covers (by decide) (by decide) row7727_covers
theorem node7727_eq_data : node7727 = Data.node7727 := by
  have branches : (node7723, node7753) = (Data.node7723, Data.node7753) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7723_eq_data node7753_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7727 row children.1 children.2) row7727_eq_data branches
set_option maxRecDepth 4096 in
theorem row7703_eq_data : row7703 = Data.row7703 := rfl
def node7703 : Graph := .node 7703 row7703 node7687 node7727
set_option maxRecDepth 4096 in
theorem node7703_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7703 :=
  Budget.node_covers _ _ 7703 row7703 node7687 node7727
    node7687_covers node7727_covers (by decide) (by decide) row7703_covers
theorem node7703_eq_data : node7703 = Data.node7703 := by
  have branches : (node7687, node7727) = (Data.node7687, Data.node7727) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7687_eq_data node7727_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7703 row children.1 children.2) row7703_eq_data branches
set_option maxRecDepth 4096 in
theorem row7759_eq_data : row7759 = Data.row7759 := rfl
def node7759 : Graph := .node 7759 row7759 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7759_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7759 :=
  Budget.node_covers _ _ 7759 row7759 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7759_covers
theorem node7759_eq_data : node7759 = Data.node7759 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7759 row children.1 children.2) row7759_eq_data branches
set_option maxRecDepth 4096 in
theorem row7789_eq_data : row7789 = Data.row7789 := rfl
def node7789 : Graph := .node 7789 row7789 node7759 Graph.empty
set_option maxRecDepth 4096 in
theorem node7789_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7789 :=
  Budget.node_covers _ _ 7789 row7789 node7759 Graph.empty
    node7759_covers empty_covers (by decide) (by decide) row7789_covers
theorem node7789_eq_data : node7789 = Data.node7789 := by
  have branches : (node7759, Graph.empty) = (Data.node7759, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7759_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7789 row children.1 children.2) row7789_eq_data branches
set_option maxRecDepth 4096 in
theorem row7817_eq_data : row7817 = Data.row7817 := rfl
def node7817 : Graph := .node 7817 row7817 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7817_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7817 :=
  Budget.node_covers _ _ 7817 row7817 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7817_covers
theorem node7817_eq_data : node7817 = Data.node7817 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7817 row children.1 children.2) row7817_eq_data branches
set_option maxRecDepth 4096 in
theorem row7823_eq_data : row7823 = Data.row7823 := rfl
def node7823 : Graph := .node 7823 row7823 node7817 Graph.empty
set_option maxRecDepth 4096 in
theorem node7823_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7823 :=
  Budget.node_covers _ _ 7823 row7823 node7817 Graph.empty
    node7817_covers empty_covers (by decide) (by decide) row7823_covers
theorem node7823_eq_data : node7823 = Data.node7823 := by
  have branches : (node7817, Graph.empty) = (Data.node7817, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7817_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7823 row children.1 children.2) row7823_eq_data branches
set_option maxRecDepth 4096 in
theorem row7793_eq_data : row7793 = Data.row7793 := rfl
def node7793 : Graph := .node 7793 row7793 node7789 node7823
set_option maxRecDepth 4096 in
theorem node7793_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7793 :=
  Budget.node_covers _ _ 7793 row7793 node7789 node7823
    node7789_covers node7823_covers (by decide) (by decide) row7793_covers
theorem node7793_eq_data : node7793 = Data.node7793 := by
  have branches : (node7789, node7823) = (Data.node7789, Data.node7823) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7789_eq_data node7823_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7793 row children.1 children.2) row7793_eq_data branches
set_option maxRecDepth 4096 in
theorem row7841_eq_data : row7841 = Data.row7841 := rfl
def node7841 : Graph := .node 7841 row7841 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7841_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7841 :=
  Budget.node_covers _ _ 7841 row7841 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7841_covers
theorem node7841_eq_data : node7841 = Data.node7841 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7841 row children.1 children.2) row7841_eq_data branches
set_option maxRecDepth 4096 in
theorem row7853_eq_data : row7853 = Data.row7853 := rfl
def node7853 : Graph := .node 7853 row7853 node7841 Graph.empty
set_option maxRecDepth 4096 in
theorem node7853_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7853 :=
  Budget.node_covers _ _ 7853 row7853 node7841 Graph.empty
    node7841_covers empty_covers (by decide) (by decide) row7853_covers
theorem node7853_eq_data : node7853 = Data.node7853 := by
  have branches : (node7841, Graph.empty) = (Data.node7841, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7841_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7853 row children.1 children.2) row7853_eq_data branches
set_option maxRecDepth 4096 in
theorem row7873_eq_data : row7873 = Data.row7873 := rfl
def node7873 : Graph := .node 7873 row7873 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7873_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7873 :=
  Budget.node_covers _ _ 7873 row7873 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7873_covers
theorem node7873_eq_data : node7873 = Data.node7873 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7873 row children.1 children.2) row7873_eq_data branches
set_option maxRecDepth 4096 in
theorem row7877_eq_data : row7877 = Data.row7877 := rfl
def node7877 : Graph := .node 7877 row7877 node7873 Graph.empty
set_option maxRecDepth 4096 in
theorem node7877_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7877 :=
  Budget.node_covers _ _ 7877 row7877 node7873 Graph.empty
    node7873_covers empty_covers (by decide) (by decide) row7877_covers
theorem node7877_eq_data : node7877 = Data.node7877 := by
  have branches : (node7873, Graph.empty) = (Data.node7873, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7873_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7877 row children.1 children.2) row7877_eq_data branches
set_option maxRecDepth 4096 in
theorem row7867_eq_data : row7867 = Data.row7867 := rfl
def node7867 : Graph := .node 7867 row7867 node7853 node7877
set_option maxRecDepth 4096 in
theorem node7867_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7867 :=
  Budget.node_covers _ _ 7867 row7867 node7853 node7877
    node7853_covers node7877_covers (by decide) (by decide) row7867_covers
theorem node7867_eq_data : node7867 = Data.node7867 := by
  have branches : (node7853, node7877) = (Data.node7853, Data.node7877) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7853_eq_data node7877_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7867 row children.1 children.2) row7867_eq_data branches
set_option maxRecDepth 4096 in
theorem row7829_eq_data : row7829 = Data.row7829 := rfl
def node7829 : Graph := .node 7829 row7829 node7793 node7867
set_option maxRecDepth 4096 in
theorem node7829_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7829 :=
  Budget.node_covers _ _ 7829 row7829 node7793 node7867
    node7793_covers node7867_covers (by decide) (by decide) row7829_covers
theorem node7829_eq_data : node7829 = Data.node7829 := by
  have branches : (node7793, node7867) = (Data.node7793, Data.node7867) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7793_eq_data node7867_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7829 row children.1 children.2) row7829_eq_data branches
set_option maxRecDepth 4096 in
theorem row7757_eq_data : row7757 = Data.row7757 := rfl
def node7757 : Graph := .node 7757 row7757 node7703 node7829
set_option maxRecDepth 4096 in
theorem node7757_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7757 :=
  Budget.node_covers _ _ 7757 row7757 node7703 node7829
    node7703_covers node7829_covers (by decide) (by decide) row7757_covers
theorem node7757_eq_data : node7757 = Data.node7757 := by
  have branches : (node7703, node7829) = (Data.node7703, Data.node7829) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7703_eq_data node7829_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7757 row children.1 children.2) row7757_eq_data branches
set_option maxRecDepth 4096 in
theorem row7883_eq_data : row7883 = Data.row7883 := rfl
def node7883 : Graph := .node 7883 row7883 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7883_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7883 :=
  Budget.node_covers _ _ 7883 row7883 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7883_covers
theorem node7883_eq_data : node7883 = Data.node7883 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7883 row children.1 children.2) row7883_eq_data branches
set_option maxRecDepth 4096 in
theorem row7901_eq_data : row7901 = Data.row7901 := rfl
def node7901 : Graph := .node 7901 row7901 node7883 Graph.empty
set_option maxRecDepth 4096 in
theorem node7901_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7901 :=
  Budget.node_covers _ _ 7901 row7901 node7883 Graph.empty
    node7883_covers empty_covers (by decide) (by decide) row7901_covers
theorem node7901_eq_data : node7901 = Data.node7901 := by
  have branches : (node7883, Graph.empty) = (Data.node7883, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7883_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7901 row children.1 children.2) row7901_eq_data branches
set_option maxRecDepth 4096 in
theorem row7919_eq_data : row7919 = Data.row7919 := rfl
def node7919 : Graph := .node 7919 row7919 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7919_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7919 :=
  Budget.node_covers _ _ 7919 row7919 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7919_covers
theorem node7919_eq_data : node7919 = Data.node7919 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7919 row children.1 children.2) row7919_eq_data branches
set_option maxRecDepth 4096 in
theorem row7927_eq_data : row7927 = Data.row7927 := rfl
def node7927 : Graph := .node 7927 row7927 node7919 Graph.empty
set_option maxRecDepth 4096 in
theorem node7927_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7927 :=
  Budget.node_covers _ _ 7927 row7927 node7919 Graph.empty
    node7919_covers empty_covers (by decide) (by decide) row7927_covers
theorem node7927_eq_data : node7927 = Data.node7927 := by
  have branches : (node7919, Graph.empty) = (Data.node7919, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7919_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7927 row children.1 children.2) row7927_eq_data branches
set_option maxRecDepth 4096 in
theorem row7907_eq_data : row7907 = Data.row7907 := rfl
def node7907 : Graph := .node 7907 row7907 node7901 node7927
set_option maxRecDepth 4096 in
theorem node7907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7907 :=
  Budget.node_covers _ _ 7907 row7907 node7901 node7927
    node7901_covers node7927_covers (by decide) (by decide) row7907_covers
theorem node7907_eq_data : node7907 = Data.node7907 := by
  have branches : (node7901, node7927) = (Data.node7901, Data.node7927) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7901_eq_data node7927_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7907 row children.1 children.2) row7907_eq_data branches
set_option maxRecDepth 4096 in
theorem row7937_eq_data : row7937 = Data.row7937 := rfl
def node7937 : Graph := .node 7937 row7937 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7937_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7937 :=
  Budget.node_covers _ _ 7937 row7937 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7937_covers
theorem node7937_eq_data : node7937 = Data.node7937 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7937 row children.1 children.2) row7937_eq_data branches
set_option maxRecDepth 4096 in
theorem row7949_eq_data : row7949 = Data.row7949 := rfl
def node7949 : Graph := .node 7949 row7949 node7937 Graph.empty
set_option maxRecDepth 4096 in
theorem node7949_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7949 :=
  Budget.node_covers _ _ 7949 row7949 node7937 Graph.empty
    node7937_covers empty_covers (by decide) (by decide) row7949_covers
theorem node7949_eq_data : node7949 = Data.node7949 := by
  have branches : (node7937, Graph.empty) = (Data.node7937, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7937_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7949 row children.1 children.2) row7949_eq_data branches
set_option maxRecDepth 4096 in
theorem row7963_eq_data : row7963 = Data.row7963 := rfl
def node7963 : Graph := .node 7963 row7963 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7963_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7963 :=
  Budget.node_covers _ _ 7963 row7963 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7963_covers
theorem node7963_eq_data : node7963 = Data.node7963 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7963 row children.1 children.2) row7963_eq_data branches
set_option maxRecDepth 4096 in
theorem row7993_eq_data : row7993 = Data.row7993 := rfl
def node7993 : Graph := .node 7993 row7993 node7963 Graph.empty
set_option maxRecDepth 4096 in
theorem node7993_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7993 :=
  Budget.node_covers _ _ 7993 row7993 node7963 Graph.empty
    node7963_covers empty_covers (by decide) (by decide) row7993_covers
theorem node7993_eq_data : node7993 = Data.node7993 := by
  have branches : (node7963, Graph.empty) = (Data.node7963, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7963_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7993 row children.1 children.2) row7993_eq_data branches
set_option maxRecDepth 4096 in
theorem row7951_eq_data : row7951 = Data.row7951 := rfl
def node7951 : Graph := .node 7951 row7951 node7949 node7993
set_option maxRecDepth 4096 in
theorem node7951_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7951 :=
  Budget.node_covers _ _ 7951 row7951 node7949 node7993
    node7949_covers node7993_covers (by decide) (by decide) row7951_covers
theorem node7951_eq_data : node7951 = Data.node7951 := by
  have branches : (node7949, node7993) = (Data.node7949, Data.node7993) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7949_eq_data node7993_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7951 row children.1 children.2) row7951_eq_data branches
set_option maxRecDepth 4096 in
theorem row7933_eq_data : row7933 = Data.row7933 := rfl
def node7933 : Graph := .node 7933 row7933 node7907 node7951
set_option maxRecDepth 4096 in
theorem node7933_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7933 :=
  Budget.node_covers _ _ 7933 row7933 node7907 node7951
    node7907_covers node7951_covers (by decide) (by decide) row7933_covers
theorem node7933_eq_data : node7933 = Data.node7933 := by
  have branches : (node7907, node7951) = (Data.node7907, Data.node7951) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7907_eq_data node7951_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7933 row children.1 children.2) row7933_eq_data branches
set_option maxRecDepth 4096 in
theorem row8011_eq_data : row8011 = Data.row8011 := rfl
def node8011 : Graph := .node 8011 row8011 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8011 :=
  Budget.node_covers _ _ 8011 row8011 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8011_covers
theorem node8011_eq_data : node8011 = Data.node8011 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8011 row children.1 children.2) row8011_eq_data branches
set_option maxRecDepth 4096 in
theorem row8017_eq_data : row8017 = Data.row8017 := rfl
def node8017 : Graph := .node 8017 row8017 node8011 Graph.empty
set_option maxRecDepth 4096 in
theorem node8017_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8017 :=
  Budget.node_covers _ _ 8017 row8017 node8011 Graph.empty
    node8011_covers empty_covers (by decide) (by decide) row8017_covers
theorem node8017_eq_data : node8017 = Data.node8017 := by
  have branches : (node8011, Graph.empty) = (Data.node8011, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8011_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8017 row children.1 children.2) row8017_eq_data branches
set_option maxRecDepth 4096 in
theorem row8053_eq_data : row8053 = Data.row8053 := rfl
def node8053 : Graph := .node 8053 row8053 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8053_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8053 :=
  Budget.node_covers _ _ 8053 row8053 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8053_covers
theorem node8053_eq_data : node8053 = Data.node8053 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8053 row children.1 children.2) row8053_eq_data branches
set_option maxRecDepth 4096 in
theorem row8059_eq_data : row8059 = Data.row8059 := rfl
def node8059 : Graph := .node 8059 row8059 node8053 Graph.empty
set_option maxRecDepth 4096 in
theorem node8059_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8059 :=
  Budget.node_covers _ _ 8059 row8059 node8053 Graph.empty
    node8053_covers empty_covers (by decide) (by decide) row8059_covers
theorem node8059_eq_data : node8059 = Data.node8059 := by
  have branches : (node8053, Graph.empty) = (Data.node8053, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8053_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8059 row children.1 children.2) row8059_eq_data branches
set_option maxRecDepth 4096 in
theorem row8039_eq_data : row8039 = Data.row8039 := rfl
def node8039 : Graph := .node 8039 row8039 node8017 node8059
set_option maxRecDepth 4096 in
theorem node8039_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8039 :=
  Budget.node_covers _ _ 8039 row8039 node8017 node8059
    node8017_covers node8059_covers (by decide) (by decide) row8039_covers
theorem node8039_eq_data : node8039 = Data.node8039 := by
  have branches : (node8017, node8059) = (Data.node8017, Data.node8059) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8017_eq_data node8059_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8039 row children.1 children.2) row8039_eq_data branches
set_option maxRecDepth 4096 in
theorem row8081_eq_data : row8081 = Data.row8081 := rfl
def node8081 : Graph := .node 8081 row8081 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8081_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8081 :=
  Budget.node_covers _ _ 8081 row8081 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8081_covers
theorem node8081_eq_data : node8081 = Data.node8081 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8081 row children.1 children.2) row8081_eq_data branches
set_option maxRecDepth 4096 in
theorem row8087_eq_data : row8087 = Data.row8087 := rfl
def node8087 : Graph := .node 8087 row8087 node8081 Graph.empty
set_option maxRecDepth 4096 in
theorem node8087_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8087 :=
  Budget.node_covers _ _ 8087 row8087 node8081 Graph.empty
    node8081_covers empty_covers (by decide) (by decide) row8087_covers
theorem node8087_eq_data : node8087 = Data.node8087 := by
  have branches : (node8081, Graph.empty) = (Data.node8081, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8081_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8087 row children.1 children.2) row8087_eq_data branches
set_option maxRecDepth 4096 in
theorem row8093_eq_data : row8093 = Data.row8093 := rfl
def node8093 : Graph := .node 8093 row8093 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8093_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8093 :=
  Budget.node_covers _ _ 8093 row8093 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8093_covers
theorem node8093_eq_data : node8093 = Data.node8093 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8093 row children.1 children.2) row8093_eq_data branches
set_option maxRecDepth 4096 in
theorem row8101_eq_data : row8101 = Data.row8101 := rfl
def node8101 : Graph := .node 8101 row8101 node8093 Graph.empty
set_option maxRecDepth 4096 in
theorem node8101_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8101 :=
  Budget.node_covers _ _ 8101 row8101 node8093 Graph.empty
    node8093_covers empty_covers (by decide) (by decide) row8101_covers
theorem node8101_eq_data : node8101 = Data.node8101 := by
  have branches : (node8093, Graph.empty) = (Data.node8093, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8093_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8101 row children.1 children.2) row8101_eq_data branches
set_option maxRecDepth 4096 in
theorem row8089_eq_data : row8089 = Data.row8089 := rfl
def node8089 : Graph := .node 8089 row8089 node8087 node8101
set_option maxRecDepth 4096 in
theorem node8089_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8089 :=
  Budget.node_covers _ _ 8089 row8089 node8087 node8101
    node8087_covers node8101_covers (by decide) (by decide) row8089_covers
theorem node8089_eq_data : node8089 = Data.node8089 := by
  have branches : (node8087, node8101) = (Data.node8087, Data.node8101) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8087_eq_data node8101_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8089 row children.1 children.2) row8089_eq_data branches
set_option maxRecDepth 4096 in
theorem row8069_eq_data : row8069 = Data.row8069 := rfl
def node8069 : Graph := .node 8069 row8069 node8039 node8089
set_option maxRecDepth 4096 in
theorem node8069_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8069 :=
  Budget.node_covers _ _ 8069 row8069 node8039 node8089
    node8039_covers node8089_covers (by decide) (by decide) row8069_covers
theorem node8069_eq_data : node8069 = Data.node8069 := by
  have branches : (node8039, node8089) = (Data.node8039, Data.node8089) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8039_eq_data node8089_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8069 row children.1 children.2) row8069_eq_data branches
set_option maxRecDepth 4096 in
theorem row8009_eq_data : row8009 = Data.row8009 := rfl
def node8009 : Graph := .node 8009 row8009 node7933 node8069
set_option maxRecDepth 4096 in
theorem node8009_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8009 :=
  Budget.node_covers _ _ 8009 row8009 node7933 node8069
    node7933_covers node8069_covers (by decide) (by decide) row8009_covers
theorem node8009_eq_data : node8009 = Data.node8009 := by
  have branches : (node7933, node8069) = (Data.node7933, Data.node8069) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7933_eq_data node8069_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8009 row children.1 children.2) row8009_eq_data branches
set_option maxRecDepth 4096 in
theorem row7879_eq_data : row7879 = Data.row7879 := rfl
def node7879 : Graph := .node 7879 row7879 node7757 node8009
set_option maxRecDepth 4096 in
theorem node7879_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7879 :=
  Budget.node_covers _ _ 7879 row7879 node7757 node8009
    node7757_covers node8009_covers (by decide) (by decide) row7879_covers
theorem node7879_eq_data : node7879 = Data.node7879 := by
  have branches : (node7757, node8009) = (Data.node7757, Data.node8009) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7757_eq_data node8009_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7879 row children.1 children.2) row7879_eq_data branches
set_option maxRecDepth 4096 in
theorem row8117_eq_data : row8117 = Data.row8117 := rfl
def node8117 : Graph := .node 8117 row8117 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8117_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8117 :=
  Budget.node_covers _ _ 8117 row8117 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8117_covers
theorem node8117_eq_data : node8117 = Data.node8117 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8117 row children.1 children.2) row8117_eq_data branches
set_option maxRecDepth 4096 in
theorem row8123_eq_data : row8123 = Data.row8123 := rfl
def node8123 : Graph := .node 8123 row8123 node8117 Graph.empty
set_option maxRecDepth 4096 in
theorem node8123_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8123 :=
  Budget.node_covers _ _ 8123 row8123 node8117 Graph.empty
    node8117_covers empty_covers (by decide) (by decide) row8123_covers
theorem node8123_eq_data : node8123 = Data.node8123 := by
  have branches : (node8117, Graph.empty) = (Data.node8117, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8117_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8123 row children.1 children.2) row8123_eq_data branches
set_option maxRecDepth 4096 in
theorem row8161_eq_data : row8161 = Data.row8161 := rfl
def node8161 : Graph := .node 8161 row8161 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8161_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8161 :=
  Budget.node_covers _ _ 8161 row8161 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8161_covers
theorem node8161_eq_data : node8161 = Data.node8161 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8161 row children.1 children.2) row8161_eq_data branches
set_option maxRecDepth 4096 in
theorem row8167_eq_data : row8167 = Data.row8167 := rfl
def node8167 : Graph := .node 8167 row8167 node8161 Graph.empty
set_option maxRecDepth 4096 in
theorem node8167_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8167 :=
  Budget.node_covers _ _ 8167 row8167 node8161 Graph.empty
    node8161_covers empty_covers (by decide) (by decide) row8167_covers
theorem node8167_eq_data : node8167 = Data.node8167 := by
  have branches : (node8161, Graph.empty) = (Data.node8161, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8161_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8167 row children.1 children.2) row8167_eq_data branches
set_option maxRecDepth 4096 in
theorem row8147_eq_data : row8147 = Data.row8147 := rfl
def node8147 : Graph := .node 8147 row8147 node8123 node8167
set_option maxRecDepth 4096 in
theorem node8147_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8147 :=
  Budget.node_covers _ _ 8147 row8147 node8123 node8167
    node8123_covers node8167_covers (by decide) (by decide) row8147_covers
theorem node8147_eq_data : node8147 = Data.node8147 := by
  have branches : (node8123, node8167) = (Data.node8123, Data.node8167) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8123_eq_data node8167_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8147 row children.1 children.2) row8147_eq_data branches
set_option maxRecDepth 4096 in
theorem row8179_eq_data : row8179 = Data.row8179 := rfl
def node8179 : Graph := .node 8179 row8179 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8179_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8179 :=
  Budget.node_covers _ _ 8179 row8179 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8179_covers
theorem node8179_eq_data : node8179 = Data.node8179 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8179 row children.1 children.2) row8179_eq_data branches
set_option maxRecDepth 4096 in
theorem row8191_eq_data : row8191 = Data.row8191 := rfl
def node8191 : Graph := .node 8191 row8191 node8179 Graph.empty
set_option maxRecDepth 4096 in
theorem node8191_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8191 :=
  Budget.node_covers _ _ 8191 row8191 node8179 Graph.empty
    node8179_covers empty_covers (by decide) (by decide) row8191_covers
theorem node8191_eq_data : node8191 = Data.node8191 := by
  have branches : (node8179, Graph.empty) = (Data.node8179, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8179_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8191 row children.1 children.2) row8191_eq_data branches
set_option maxRecDepth 4096 in
theorem row8219_eq_data : row8219 = Data.row8219 := rfl
def node8219 : Graph := .node 8219 row8219 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8219_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8219 :=
  Budget.node_covers _ _ 8219 row8219 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8219_covers
theorem node8219_eq_data : node8219 = Data.node8219 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8219 row children.1 children.2) row8219_eq_data branches
set_option maxRecDepth 4096 in
theorem row8221_eq_data : row8221 = Data.row8221 := rfl
def node8221 : Graph := .node 8221 row8221 node8219 Graph.empty
set_option maxRecDepth 4096 in
theorem node8221_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8221 :=
  Budget.node_covers _ _ 8221 row8221 node8219 Graph.empty
    node8219_covers empty_covers (by decide) (by decide) row8221_covers
theorem node8221_eq_data : node8221 = Data.node8221 := by
  have branches : (node8219, Graph.empty) = (Data.node8219, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8219_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8221 row children.1 children.2) row8221_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
