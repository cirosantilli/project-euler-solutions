import ProjectEulerSolutions.Lib.PrimePairLarge.Graph005
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row2297_eq_data : row2297 = Data.row2297 := rfl
def node2297 : Graph := .node 2297 row2297 node2111 node2473
set_option maxRecDepth 4096 in
theorem node2297_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2297 :=
  Budget.node_covers _ _ 2297 row2297 node2111 node2473
    node2111_covers node2473_covers (by decide) (by decide) row2297_covers
theorem node2297_eq_data : node2297 = Data.node2297 := by
  have branches : (node2111, node2473) = (Data.node2111, Data.node2473) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2111_eq_data node2473_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2297 row children.1 children.2) row2297_eq_data branches
set_option maxRecDepth 4096 in
theorem row1913_eq_data : row1913 = Data.row1913 := rfl
def node1913 : Graph := .node 1913 row1913 node1553 node2297
set_option maxRecDepth 4096 in
theorem node1913_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1913 :=
  Budget.node_covers _ _ 1913 row1913 node1553 node2297
    node1553_covers node2297_covers (by decide) (by decide) row1913_covers
theorem node1913_eq_data : node1913 = Data.node1913 := by
  have branches : (node1553, node2297) = (Data.node1553, Data.node2297) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1553_eq_data node2297_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1913 row children.1 children.2) row1913_eq_data branches
set_option maxRecDepth 4096 in
theorem row1193_eq_data : row1193 = Data.row1193 := rfl
def node1193 : Graph := .node 1193 row1193 node523 node1913
set_option maxRecDepth 4096 in
theorem node1193_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1193 :=
  Budget.node_covers _ _ 1193 row1193 node523 node1913
    node523_covers node1913_covers (by decide) (by decide) row1193_covers
theorem node1193_eq_data : node1193 = Data.node1193 := by
  have branches : (node523, node1913) = (Data.node523, Data.node1913) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node523_eq_data node1913_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1193 row children.1 children.2) row1193_eq_data branches
set_option maxRecDepth 4096 in
theorem row2689_eq_data : row2689 = Data.row2689 := rfl
def node2689 : Graph := .node 2689 row2689 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2689 :=
  Budget.node_covers _ _ 2689 row2689 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2689_covers
theorem node2689_eq_data : node2689 = Data.node2689 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2689 row children.1 children.2) row2689_eq_data branches
set_option maxRecDepth 4096 in
theorem row2699_eq_data : row2699 = Data.row2699 := rfl
def node2699 : Graph := .node 2699 row2699 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2699_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2699 :=
  Budget.node_covers _ _ 2699 row2699 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2699_covers
theorem node2699_eq_data : node2699 = Data.node2699 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2699 row children.1 children.2) row2699_eq_data branches
set_option maxRecDepth 4096 in
theorem row2693_eq_data : row2693 = Data.row2693 := rfl
def node2693 : Graph := .node 2693 row2693 node2689 node2699
set_option maxRecDepth 4096 in
theorem node2693_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2693 :=
  Budget.node_covers _ _ 2693 row2693 node2689 node2699
    node2689_covers node2699_covers (by decide) (by decide) row2693_covers
theorem node2693_eq_data : node2693 = Data.node2693 := by
  have branches : (node2689, node2699) = (Data.node2689, Data.node2699) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2689_eq_data node2699_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2693 row children.1 children.2) row2693_eq_data branches
set_option maxRecDepth 4096 in
theorem row2711_eq_data : row2711 = Data.row2711 := rfl
def node2711 : Graph := .node 2711 row2711 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2711_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2711 :=
  Budget.node_covers _ _ 2711 row2711 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2711_covers
theorem node2711_eq_data : node2711 = Data.node2711 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2711 row children.1 children.2) row2711_eq_data branches
set_option maxRecDepth 4096 in
theorem row2713_eq_data : row2713 = Data.row2713 := rfl
def node2713 : Graph := .node 2713 row2713 node2711 Graph.empty
set_option maxRecDepth 4096 in
theorem node2713_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2713 :=
  Budget.node_covers _ _ 2713 row2713 node2711 Graph.empty
    node2711_covers empty_covers (by decide) (by decide) row2713_covers
theorem node2713_eq_data : node2713 = Data.node2713 := by
  have branches : (node2711, Graph.empty) = (Data.node2711, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2711_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2713 row children.1 children.2) row2713_eq_data branches
set_option maxRecDepth 4096 in
theorem row2707_eq_data : row2707 = Data.row2707 := rfl
def node2707 : Graph := .node 2707 row2707 node2693 node2713
set_option maxRecDepth 4096 in
theorem node2707_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2707 :=
  Budget.node_covers _ _ 2707 row2707 node2693 node2713
    node2693_covers node2713_covers (by decide) (by decide) row2707_covers
theorem node2707_eq_data : node2707 = Data.node2707 := by
  have branches : (node2693, node2713) = (Data.node2693, Data.node2713) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2693_eq_data node2713_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2707 row children.1 children.2) row2707_eq_data branches
set_option maxRecDepth 4096 in
theorem row2729_eq_data : row2729 = Data.row2729 := rfl
def node2729 : Graph := .node 2729 row2729 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2729_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2729 :=
  Budget.node_covers _ _ 2729 row2729 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2729_covers
theorem node2729_eq_data : node2729 = Data.node2729 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2729 row children.1 children.2) row2729_eq_data branches
set_option maxRecDepth 4096 in
theorem row2731_eq_data : row2731 = Data.row2731 := rfl
def node2731 : Graph := .node 2731 row2731 node2729 Graph.empty
set_option maxRecDepth 4096 in
theorem node2731_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2731 :=
  Budget.node_covers _ _ 2731 row2731 node2729 Graph.empty
    node2729_covers empty_covers (by decide) (by decide) row2731_covers
theorem node2731_eq_data : node2731 = Data.node2731 := by
  have branches : (node2729, Graph.empty) = (Data.node2729, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2729_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2731 row children.1 children.2) row2731_eq_data branches
set_option maxRecDepth 4096 in
theorem row2749_eq_data : row2749 = Data.row2749 := rfl
def node2749 : Graph := .node 2749 row2749 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2749_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2749 :=
  Budget.node_covers _ _ 2749 row2749 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2749_covers
theorem node2749_eq_data : node2749 = Data.node2749 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2749 row children.1 children.2) row2749_eq_data branches
set_option maxRecDepth 4096 in
theorem row2753_eq_data : row2753 = Data.row2753 := rfl
def node2753 : Graph := .node 2753 row2753 node2749 Graph.empty
set_option maxRecDepth 4096 in
theorem node2753_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2753 :=
  Budget.node_covers _ _ 2753 row2753 node2749 Graph.empty
    node2749_covers empty_covers (by decide) (by decide) row2753_covers
theorem node2753_eq_data : node2753 = Data.node2753 := by
  have branches : (node2749, Graph.empty) = (Data.node2749, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2749_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2753 row children.1 children.2) row2753_eq_data branches
set_option maxRecDepth 4096 in
theorem row2741_eq_data : row2741 = Data.row2741 := rfl
def node2741 : Graph := .node 2741 row2741 node2731 node2753
set_option maxRecDepth 4096 in
theorem node2741_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2741 :=
  Budget.node_covers _ _ 2741 row2741 node2731 node2753
    node2731_covers node2753_covers (by decide) (by decide) row2741_covers
theorem node2741_eq_data : node2741 = Data.node2741 := by
  have branches : (node2731, node2753) = (Data.node2731, Data.node2753) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2731_eq_data node2753_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2741 row children.1 children.2) row2741_eq_data branches
set_option maxRecDepth 4096 in
theorem row2719_eq_data : row2719 = Data.row2719 := rfl
def node2719 : Graph := .node 2719 row2719 node2707 node2741
set_option maxRecDepth 4096 in
theorem node2719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2719 :=
  Budget.node_covers _ _ 2719 row2719 node2707 node2741
    node2707_covers node2741_covers (by decide) (by decide) row2719_covers
theorem node2719_eq_data : node2719 = Data.node2719 := by
  have branches : (node2707, node2741) = (Data.node2707, Data.node2741) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2707_eq_data node2741_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2719 row children.1 children.2) row2719_eq_data branches
set_option maxRecDepth 4096 in
theorem row2777_eq_data : row2777 = Data.row2777 := rfl
def node2777 : Graph := .node 2777 row2777 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2777_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2777 :=
  Budget.node_covers _ _ 2777 row2777 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2777_covers
theorem node2777_eq_data : node2777 = Data.node2777 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2777 row children.1 children.2) row2777_eq_data branches
set_option maxRecDepth 4096 in
theorem row2789_eq_data : row2789 = Data.row2789 := rfl
def node2789 : Graph := .node 2789 row2789 node2777 Graph.empty
set_option maxRecDepth 4096 in
theorem node2789_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2789 :=
  Budget.node_covers _ _ 2789 row2789 node2777 Graph.empty
    node2777_covers empty_covers (by decide) (by decide) row2789_covers
theorem node2789_eq_data : node2789 = Data.node2789 := by
  have branches : (node2777, Graph.empty) = (Data.node2777, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2777_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2789 row children.1 children.2) row2789_eq_data branches
set_option maxRecDepth 4096 in
theorem row2797_eq_data : row2797 = Data.row2797 := rfl
def node2797 : Graph := .node 2797 row2797 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2797_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2797 :=
  Budget.node_covers _ _ 2797 row2797 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2797_covers
theorem node2797_eq_data : node2797 = Data.node2797 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2797 row children.1 children.2) row2797_eq_data branches
set_option maxRecDepth 4096 in
theorem row2801_eq_data : row2801 = Data.row2801 := rfl
def node2801 : Graph := .node 2801 row2801 node2797 Graph.empty
set_option maxRecDepth 4096 in
theorem node2801_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2801 :=
  Budget.node_covers _ _ 2801 row2801 node2797 Graph.empty
    node2797_covers empty_covers (by decide) (by decide) row2801_covers
theorem node2801_eq_data : node2801 = Data.node2801 := by
  have branches : (node2797, Graph.empty) = (Data.node2797, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2797_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2801 row children.1 children.2) row2801_eq_data branches
set_option maxRecDepth 4096 in
theorem row2791_eq_data : row2791 = Data.row2791 := rfl
def node2791 : Graph := .node 2791 row2791 node2789 node2801
set_option maxRecDepth 4096 in
theorem node2791_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2791 :=
  Budget.node_covers _ _ 2791 row2791 node2789 node2801
    node2789_covers node2801_covers (by decide) (by decide) row2791_covers
theorem node2791_eq_data : node2791 = Data.node2791 := by
  have branches : (node2789, node2801) = (Data.node2789, Data.node2801) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2789_eq_data node2801_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2791 row children.1 children.2) row2791_eq_data branches
set_option maxRecDepth 4096 in
theorem row2819_eq_data : row2819 = Data.row2819 := rfl
def node2819 : Graph := .node 2819 row2819 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2819_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2819 :=
  Budget.node_covers _ _ 2819 row2819 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2819_covers
theorem node2819_eq_data : node2819 = Data.node2819 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2819 row children.1 children.2) row2819_eq_data branches
set_option maxRecDepth 4096 in
theorem row2833_eq_data : row2833 = Data.row2833 := rfl
def node2833 : Graph := .node 2833 row2833 node2819 Graph.empty
set_option maxRecDepth 4096 in
theorem node2833_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2833 :=
  Budget.node_covers _ _ 2833 row2833 node2819 Graph.empty
    node2819_covers empty_covers (by decide) (by decide) row2833_covers
theorem node2833_eq_data : node2833 = Data.node2833 := by
  have branches : (node2819, Graph.empty) = (Data.node2819, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2819_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2833 row children.1 children.2) row2833_eq_data branches
set_option maxRecDepth 4096 in
theorem row2843_eq_data : row2843 = Data.row2843 := rfl
def node2843 : Graph := .node 2843 row2843 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2843_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2843 :=
  Budget.node_covers _ _ 2843 row2843 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2843_covers
theorem node2843_eq_data : node2843 = Data.node2843 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2843 row children.1 children.2) row2843_eq_data branches
set_option maxRecDepth 4096 in
theorem row2851_eq_data : row2851 = Data.row2851 := rfl
def node2851 : Graph := .node 2851 row2851 node2843 Graph.empty
set_option maxRecDepth 4096 in
theorem node2851_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2851 :=
  Budget.node_covers _ _ 2851 row2851 node2843 Graph.empty
    node2843_covers empty_covers (by decide) (by decide) row2851_covers
theorem node2851_eq_data : node2851 = Data.node2851 := by
  have branches : (node2843, Graph.empty) = (Data.node2843, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2843_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2851 row children.1 children.2) row2851_eq_data branches
set_option maxRecDepth 4096 in
theorem row2837_eq_data : row2837 = Data.row2837 := rfl
def node2837 : Graph := .node 2837 row2837 node2833 node2851
set_option maxRecDepth 4096 in
theorem node2837_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2837 :=
  Budget.node_covers _ _ 2837 row2837 node2833 node2851
    node2833_covers node2851_covers (by decide) (by decide) row2837_covers
theorem node2837_eq_data : node2837 = Data.node2837 := by
  have branches : (node2833, node2851) = (Data.node2833, Data.node2851) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2833_eq_data node2851_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2837 row children.1 children.2) row2837_eq_data branches
set_option maxRecDepth 4096 in
theorem row2803_eq_data : row2803 = Data.row2803 := rfl
def node2803 : Graph := .node 2803 row2803 node2791 node2837
set_option maxRecDepth 4096 in
theorem node2803_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2803 :=
  Budget.node_covers _ _ 2803 row2803 node2791 node2837
    node2791_covers node2837_covers (by decide) (by decide) row2803_covers
theorem node2803_eq_data : node2803 = Data.node2803 := by
  have branches : (node2791, node2837) = (Data.node2791, Data.node2837) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2791_eq_data node2837_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2803 row children.1 children.2) row2803_eq_data branches
set_option maxRecDepth 4096 in
theorem row2767_eq_data : row2767 = Data.row2767 := rfl
def node2767 : Graph := .node 2767 row2767 node2719 node2803
set_option maxRecDepth 4096 in
theorem node2767_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2767 :=
  Budget.node_covers _ _ 2767 row2767 node2719 node2803
    node2719_covers node2803_covers (by decide) (by decide) row2767_covers
theorem node2767_eq_data : node2767 = Data.node2767 := by
  have branches : (node2719, node2803) = (Data.node2719, Data.node2803) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2719_eq_data node2803_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2767 row children.1 children.2) row2767_eq_data branches
set_option maxRecDepth 4096 in
theorem row2861_eq_data : row2861 = Data.row2861 := rfl
def node2861 : Graph := .node 2861 row2861 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2861_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2861 :=
  Budget.node_covers _ _ 2861 row2861 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2861_covers
theorem node2861_eq_data : node2861 = Data.node2861 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2861 row children.1 children.2) row2861_eq_data branches
set_option maxRecDepth 4096 in
theorem row2879_eq_data : row2879 = Data.row2879 := rfl
def node2879 : Graph := .node 2879 row2879 node2861 Graph.empty
set_option maxRecDepth 4096 in
theorem node2879_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2879 :=
  Budget.node_covers _ _ 2879 row2879 node2861 Graph.empty
    node2861_covers empty_covers (by decide) (by decide) row2879_covers
theorem node2879_eq_data : node2879 = Data.node2879 := by
  have branches : (node2861, Graph.empty) = (Data.node2861, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2861_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2879 row children.1 children.2) row2879_eq_data branches
set_option maxRecDepth 4096 in
theorem row2897_eq_data : row2897 = Data.row2897 := rfl
def node2897 : Graph := .node 2897 row2897 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2897_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2897 :=
  Budget.node_covers _ _ 2897 row2897 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2897_covers
theorem node2897_eq_data : node2897 = Data.node2897 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2897 row children.1 children.2) row2897_eq_data branches
set_option maxRecDepth 4096 in
theorem row2903_eq_data : row2903 = Data.row2903 := rfl
def node2903 : Graph := .node 2903 row2903 node2897 Graph.empty
set_option maxRecDepth 4096 in
theorem node2903_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2903 :=
  Budget.node_covers _ _ 2903 row2903 node2897 Graph.empty
    node2897_covers empty_covers (by decide) (by decide) row2903_covers
theorem node2903_eq_data : node2903 = Data.node2903 := by
  have branches : (node2897, Graph.empty) = (Data.node2897, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2897_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2903 row children.1 children.2) row2903_eq_data branches
set_option maxRecDepth 4096 in
theorem row2887_eq_data : row2887 = Data.row2887 := rfl
def node2887 : Graph := .node 2887 row2887 node2879 node2903
set_option maxRecDepth 4096 in
theorem node2887_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2887 :=
  Budget.node_covers _ _ 2887 row2887 node2879 node2903
    node2879_covers node2903_covers (by decide) (by decide) row2887_covers
theorem node2887_eq_data : node2887 = Data.node2887 := by
  have branches : (node2879, node2903) = (Data.node2879, Data.node2903) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2879_eq_data node2903_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2887 row children.1 children.2) row2887_eq_data branches
set_option maxRecDepth 4096 in
theorem row2917_eq_data : row2917 = Data.row2917 := rfl
def node2917 : Graph := .node 2917 row2917 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2917_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2917 :=
  Budget.node_covers _ _ 2917 row2917 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2917_covers
theorem node2917_eq_data : node2917 = Data.node2917 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2917 row children.1 children.2) row2917_eq_data branches
set_option maxRecDepth 4096 in
theorem row2927_eq_data : row2927 = Data.row2927 := rfl
def node2927 : Graph := .node 2927 row2927 node2917 Graph.empty
set_option maxRecDepth 4096 in
theorem node2927_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2927 :=
  Budget.node_covers _ _ 2927 row2927 node2917 Graph.empty
    node2917_covers empty_covers (by decide) (by decide) row2927_covers
theorem node2927_eq_data : node2927 = Data.node2927 := by
  have branches : (node2917, Graph.empty) = (Data.node2917, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2917_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2927 row children.1 children.2) row2927_eq_data branches
set_option maxRecDepth 4096 in
theorem row2953_eq_data : row2953 = Data.row2953 := rfl
def node2953 : Graph := .node 2953 row2953 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2953_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2953 :=
  Budget.node_covers _ _ 2953 row2953 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2953_covers
theorem node2953_eq_data : node2953 = Data.node2953 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2953 row children.1 children.2) row2953_eq_data branches
set_option maxRecDepth 4096 in
theorem row2957_eq_data : row2957 = Data.row2957 := rfl
def node2957 : Graph := .node 2957 row2957 node2953 Graph.empty
set_option maxRecDepth 4096 in
theorem node2957_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2957 :=
  Budget.node_covers _ _ 2957 row2957 node2953 Graph.empty
    node2953_covers empty_covers (by decide) (by decide) row2957_covers
theorem node2957_eq_data : node2957 = Data.node2957 := by
  have branches : (node2953, Graph.empty) = (Data.node2953, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2953_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2957 row children.1 children.2) row2957_eq_data branches
set_option maxRecDepth 4096 in
theorem row2939_eq_data : row2939 = Data.row2939 := rfl
def node2939 : Graph := .node 2939 row2939 node2927 node2957
set_option maxRecDepth 4096 in
theorem node2939_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2939 :=
  Budget.node_covers _ _ 2939 row2939 node2927 node2957
    node2927_covers node2957_covers (by decide) (by decide) row2939_covers
theorem node2939_eq_data : node2939 = Data.node2939 := by
  have branches : (node2927, node2957) = (Data.node2927, Data.node2957) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2927_eq_data node2957_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2939 row children.1 children.2) row2939_eq_data branches
set_option maxRecDepth 4096 in
theorem row2909_eq_data : row2909 = Data.row2909 := rfl
def node2909 : Graph := .node 2909 row2909 node2887 node2939
set_option maxRecDepth 4096 in
theorem node2909_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2909 :=
  Budget.node_covers _ _ 2909 row2909 node2887 node2939
    node2887_covers node2939_covers (by decide) (by decide) row2909_covers
theorem node2909_eq_data : node2909 = Data.node2909 := by
  have branches : (node2887, node2939) = (Data.node2887, Data.node2939) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2887_eq_data node2939_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2909 row children.1 children.2) row2909_eq_data branches
set_option maxRecDepth 4096 in
theorem row2969_eq_data : row2969 = Data.row2969 := rfl
def node2969 : Graph := .node 2969 row2969 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2969_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2969 :=
  Budget.node_covers _ _ 2969 row2969 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2969_covers
theorem node2969_eq_data : node2969 = Data.node2969 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2969 row children.1 children.2) row2969_eq_data branches
set_option maxRecDepth 4096 in
theorem row2971_eq_data : row2971 = Data.row2971 := rfl
def node2971 : Graph := .node 2971 row2971 node2969 Graph.empty
set_option maxRecDepth 4096 in
theorem node2971_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2971 :=
  Budget.node_covers _ _ 2971 row2971 node2969 Graph.empty
    node2969_covers empty_covers (by decide) (by decide) row2971_covers
theorem node2971_eq_data : node2971 = Data.node2971 := by
  have branches : (node2969, Graph.empty) = (Data.node2969, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2969_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2971 row children.1 children.2) row2971_eq_data branches
set_option maxRecDepth 4096 in
theorem row3001_eq_data : row3001 = Data.row3001 := rfl
def node3001 : Graph := .node 3001 row3001 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3001_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3001 :=
  Budget.node_covers _ _ 3001 row3001 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3001_covers
theorem node3001_eq_data : node3001 = Data.node3001 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3001 row children.1 children.2) row3001_eq_data branches
set_option maxRecDepth 4096 in
theorem row3011_eq_data : row3011 = Data.row3011 := rfl
def node3011 : Graph := .node 3011 row3011 node3001 Graph.empty
set_option maxRecDepth 4096 in
theorem node3011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3011 :=
  Budget.node_covers _ _ 3011 row3011 node3001 Graph.empty
    node3001_covers empty_covers (by decide) (by decide) row3011_covers
theorem node3011_eq_data : node3011 = Data.node3011 := by
  have branches : (node3001, Graph.empty) = (Data.node3001, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3001_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3011 row children.1 children.2) row3011_eq_data branches
set_option maxRecDepth 4096 in
theorem row2999_eq_data : row2999 = Data.row2999 := rfl
def node2999 : Graph := .node 2999 row2999 node2971 node3011
set_option maxRecDepth 4096 in
theorem node2999_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2999 :=
  Budget.node_covers _ _ 2999 row2999 node2971 node3011
    node2971_covers node3011_covers (by decide) (by decide) row2999_covers
theorem node2999_eq_data : node2999 = Data.node2999 := by
  have branches : (node2971, node3011) = (Data.node2971, Data.node3011) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2971_eq_data node3011_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2999 row children.1 children.2) row2999_eq_data branches
set_option maxRecDepth 4096 in
theorem row3023_eq_data : row3023 = Data.row3023 := rfl
def node3023 : Graph := .node 3023 row3023 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3023_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3023 :=
  Budget.node_covers _ _ 3023 row3023 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3023_covers
theorem node3023_eq_data : node3023 = Data.node3023 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3023 row children.1 children.2) row3023_eq_data branches
set_option maxRecDepth 4096 in
theorem row3037_eq_data : row3037 = Data.row3037 := rfl
def node3037 : Graph := .node 3037 row3037 node3023 Graph.empty
set_option maxRecDepth 4096 in
theorem node3037_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3037 :=
  Budget.node_covers _ _ 3037 row3037 node3023 Graph.empty
    node3023_covers empty_covers (by decide) (by decide) row3037_covers
theorem node3037_eq_data : node3037 = Data.node3037 := by
  have branches : (node3023, Graph.empty) = (Data.node3023, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3023_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3037 row children.1 children.2) row3037_eq_data branches
set_option maxRecDepth 4096 in
theorem row3049_eq_data : row3049 = Data.row3049 := rfl
def node3049 : Graph := .node 3049 row3049 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3049_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3049 :=
  Budget.node_covers _ _ 3049 row3049 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3049_covers
theorem node3049_eq_data : node3049 = Data.node3049 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3049 row children.1 children.2) row3049_eq_data branches
set_option maxRecDepth 4096 in
theorem row3061_eq_data : row3061 = Data.row3061 := rfl
def node3061 : Graph := .node 3061 row3061 node3049 Graph.empty
set_option maxRecDepth 4096 in
theorem node3061_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3061 :=
  Budget.node_covers _ _ 3061 row3061 node3049 Graph.empty
    node3049_covers empty_covers (by decide) (by decide) row3061_covers
theorem node3061_eq_data : node3061 = Data.node3061 := by
  have branches : (node3049, Graph.empty) = (Data.node3049, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3049_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3061 row children.1 children.2) row3061_eq_data branches
set_option maxRecDepth 4096 in
theorem row3041_eq_data : row3041 = Data.row3041 := rfl
def node3041 : Graph := .node 3041 row3041 node3037 node3061
set_option maxRecDepth 4096 in
theorem node3041_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3041 :=
  Budget.node_covers _ _ 3041 row3041 node3037 node3061
    node3037_covers node3061_covers (by decide) (by decide) row3041_covers
theorem node3041_eq_data : node3041 = Data.node3041 := by
  have branches : (node3037, node3061) = (Data.node3037, Data.node3061) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3037_eq_data node3061_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3041 row children.1 children.2) row3041_eq_data branches
set_option maxRecDepth 4096 in
theorem row3019_eq_data : row3019 = Data.row3019 := rfl
def node3019 : Graph := .node 3019 row3019 node2999 node3041
set_option maxRecDepth 4096 in
theorem node3019_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3019 :=
  Budget.node_covers _ _ 3019 row3019 node2999 node3041
    node2999_covers node3041_covers (by decide) (by decide) row3019_covers
theorem node3019_eq_data : node3019 = Data.node3019 := by
  have branches : (node2999, node3041) = (Data.node2999, Data.node3041) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2999_eq_data node3041_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3019 row children.1 children.2) row3019_eq_data branches
set_option maxRecDepth 4096 in
theorem row2963_eq_data : row2963 = Data.row2963 := rfl
def node2963 : Graph := .node 2963 row2963 node2909 node3019
set_option maxRecDepth 4096 in
theorem node2963_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2963 :=
  Budget.node_covers _ _ 2963 row2963 node2909 node3019
    node2909_covers node3019_covers (by decide) (by decide) row2963_covers
theorem node2963_eq_data : node2963 = Data.node2963 := by
  have branches : (node2909, node3019) = (Data.node2909, Data.node3019) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2909_eq_data node3019_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2963 row children.1 children.2) row2963_eq_data branches
set_option maxRecDepth 4096 in
theorem row2857_eq_data : row2857 = Data.row2857 := rfl
def node2857 : Graph := .node 2857 row2857 node2767 node2963
set_option maxRecDepth 4096 in
theorem node2857_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2857 :=
  Budget.node_covers _ _ 2857 row2857 node2767 node2963
    node2767_covers node2963_covers (by decide) (by decide) row2857_covers
theorem node2857_eq_data : node2857 = Data.node2857 := by
  have branches : (node2767, node2963) = (Data.node2767, Data.node2963) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2767_eq_data node2963_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2857 row children.1 children.2) row2857_eq_data branches
set_option maxRecDepth 4096 in
theorem row3079_eq_data : row3079 = Data.row3079 := rfl
def node3079 : Graph := .node 3079 row3079 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3079_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3079 :=
  Budget.node_covers _ _ 3079 row3079 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3079_covers
theorem node3079_eq_data : node3079 = Data.node3079 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3079 row children.1 children.2) row3079_eq_data branches
set_option maxRecDepth 4096 in
theorem row3083_eq_data : row3083 = Data.row3083 := rfl
def node3083 : Graph := .node 3083 row3083 node3079 Graph.empty
set_option maxRecDepth 4096 in
theorem node3083_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3083 :=
  Budget.node_covers _ _ 3083 row3083 node3079 Graph.empty
    node3079_covers empty_covers (by decide) (by decide) row3083_covers
theorem node3083_eq_data : node3083 = Data.node3083 := by
  have branches : (node3079, Graph.empty) = (Data.node3079, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3079_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3083 row children.1 children.2) row3083_eq_data branches
set_option maxRecDepth 4096 in
theorem row3109_eq_data : row3109 = Data.row3109 := rfl
def node3109 : Graph := .node 3109 row3109 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3109_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3109 :=
  Budget.node_covers _ _ 3109 row3109 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3109_covers
theorem node3109_eq_data : node3109 = Data.node3109 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3109 row children.1 children.2) row3109_eq_data branches
set_option maxRecDepth 4096 in
theorem row3119_eq_data : row3119 = Data.row3119 := rfl
def node3119 : Graph := .node 3119 row3119 node3109 Graph.empty
set_option maxRecDepth 4096 in
theorem node3119_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3119 :=
  Budget.node_covers _ _ 3119 row3119 node3109 Graph.empty
    node3109_covers empty_covers (by decide) (by decide) row3119_covers
theorem node3119_eq_data : node3119 = Data.node3119 := by
  have branches : (node3109, Graph.empty) = (Data.node3109, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3109_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3119 row children.1 children.2) row3119_eq_data branches
set_option maxRecDepth 4096 in
theorem row3089_eq_data : row3089 = Data.row3089 := rfl
def node3089 : Graph := .node 3089 row3089 node3083 node3119
set_option maxRecDepth 4096 in
theorem node3089_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3089 :=
  Budget.node_covers _ _ 3089 row3089 node3083 node3119
    node3083_covers node3119_covers (by decide) (by decide) row3089_covers
theorem node3089_eq_data : node3089 = Data.node3089 := by
  have branches : (node3083, node3119) = (Data.node3083, Data.node3119) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3083_eq_data node3119_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3089 row children.1 children.2) row3089_eq_data branches
set_option maxRecDepth 4096 in
theorem row3137_eq_data : row3137 = Data.row3137 := rfl
def node3137 : Graph := .node 3137 row3137 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3137_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3137 :=
  Budget.node_covers _ _ 3137 row3137 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3137_covers
theorem node3137_eq_data : node3137 = Data.node3137 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3137 row children.1 children.2) row3137_eq_data branches
set_option maxRecDepth 4096 in
theorem row3163_eq_data : row3163 = Data.row3163 := rfl
def node3163 : Graph := .node 3163 row3163 node3137 Graph.empty
set_option maxRecDepth 4096 in
theorem node3163_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3163 :=
  Budget.node_covers _ _ 3163 row3163 node3137 Graph.empty
    node3137_covers empty_covers (by decide) (by decide) row3163_covers
theorem node3163_eq_data : node3163 = Data.node3163 := by
  have branches : (node3137, Graph.empty) = (Data.node3137, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3137_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3163 row children.1 children.2) row3163_eq_data branches
set_option maxRecDepth 4096 in
theorem row3169_eq_data : row3169 = Data.row3169 := rfl
def node3169 : Graph := .node 3169 row3169 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3169_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3169 :=
  Budget.node_covers _ _ 3169 row3169 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3169_covers
theorem node3169_eq_data : node3169 = Data.node3169 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3169 row children.1 children.2) row3169_eq_data branches
set_option maxRecDepth 4096 in
theorem row3181_eq_data : row3181 = Data.row3181 := rfl
def node3181 : Graph := .node 3181 row3181 node3169 Graph.empty
set_option maxRecDepth 4096 in
theorem node3181_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3181 :=
  Budget.node_covers _ _ 3181 row3181 node3169 Graph.empty
    node3169_covers empty_covers (by decide) (by decide) row3181_covers
theorem node3181_eq_data : node3181 = Data.node3181 := by
  have branches : (node3169, Graph.empty) = (Data.node3169, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3169_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3181 row children.1 children.2) row3181_eq_data branches
set_option maxRecDepth 4096 in
theorem row3167_eq_data : row3167 = Data.row3167 := rfl
def node3167 : Graph := .node 3167 row3167 node3163 node3181
set_option maxRecDepth 4096 in
theorem node3167_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3167 :=
  Budget.node_covers _ _ 3167 row3167 node3163 node3181
    node3163_covers node3181_covers (by decide) (by decide) row3167_covers
theorem node3167_eq_data : node3167 = Data.node3167 := by
  have branches : (node3163, node3181) = (Data.node3163, Data.node3181) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3163_eq_data node3181_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3167 row children.1 children.2) row3167_eq_data branches
set_option maxRecDepth 4096 in
theorem row3121_eq_data : row3121 = Data.row3121 := rfl
def node3121 : Graph := .node 3121 row3121 node3089 node3167
set_option maxRecDepth 4096 in
theorem node3121_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3121 :=
  Budget.node_covers _ _ 3121 row3121 node3089 node3167
    node3089_covers node3167_covers (by decide) (by decide) row3121_covers
theorem node3121_eq_data : node3121 = Data.node3121 := by
  have branches : (node3089, node3167) = (Data.node3089, Data.node3167) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3089_eq_data node3167_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3121 row children.1 children.2) row3121_eq_data branches
set_option maxRecDepth 4096 in
theorem row3191_eq_data : row3191 = Data.row3191 := rfl
def node3191 : Graph := .node 3191 row3191 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3191_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3191 :=
  Budget.node_covers _ _ 3191 row3191 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3191_covers
theorem node3191_eq_data : node3191 = Data.node3191 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3191 row children.1 children.2) row3191_eq_data branches
set_option maxRecDepth 4096 in
theorem row3203_eq_data : row3203 = Data.row3203 := rfl
def node3203 : Graph := .node 3203 row3203 node3191 Graph.empty
set_option maxRecDepth 4096 in
theorem node3203_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3203 :=
  Budget.node_covers _ _ 3203 row3203 node3191 Graph.empty
    node3191_covers empty_covers (by decide) (by decide) row3203_covers
theorem node3203_eq_data : node3203 = Data.node3203 := by
  have branches : (node3191, Graph.empty) = (Data.node3191, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3191_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3203 row children.1 children.2) row3203_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
