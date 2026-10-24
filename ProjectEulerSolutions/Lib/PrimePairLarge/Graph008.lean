import ProjectEulerSolutions.Lib.PrimePairLarge.Graph007
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row3719_eq_data : row3719 = Data.row3719 := rfl
def node3719 : Graph := .node 3719 row3719 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3719 :=
  Budget.node_covers _ _ 3719 row3719 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3719_covers
theorem node3719_eq_data : node3719 = Data.node3719 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3719 row children.1 children.2) row3719_eq_data branches
set_option maxRecDepth 4096 in
theorem row3727_eq_data : row3727 = Data.row3727 := rfl
def node3727 : Graph := .node 3727 row3727 node3719 Graph.empty
set_option maxRecDepth 4096 in
theorem node3727_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3727 :=
  Budget.node_covers _ _ 3727 row3727 node3719 Graph.empty
    node3719_covers empty_covers (by decide) (by decide) row3727_covers
theorem node3727_eq_data : node3727 = Data.node3727 := by
  have branches : (node3719, Graph.empty) = (Data.node3719, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3719_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3727 row children.1 children.2) row3727_eq_data branches
set_option maxRecDepth 4096 in
theorem row3739_eq_data : row3739 = Data.row3739 := rfl
def node3739 : Graph := .node 3739 row3739 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3739_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3739 :=
  Budget.node_covers _ _ 3739 row3739 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3739_covers
theorem node3739_eq_data : node3739 = Data.node3739 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3739 row children.1 children.2) row3739_eq_data branches
set_option maxRecDepth 4096 in
theorem row3761_eq_data : row3761 = Data.row3761 := rfl
def node3761 : Graph := .node 3761 row3761 node3739 Graph.empty
set_option maxRecDepth 4096 in
theorem node3761_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3761 :=
  Budget.node_covers _ _ 3761 row3761 node3739 Graph.empty
    node3739_covers empty_covers (by decide) (by decide) row3761_covers
theorem node3761_eq_data : node3761 = Data.node3761 := by
  have branches : (node3739, Graph.empty) = (Data.node3739, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3739_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3761 row children.1 children.2) row3761_eq_data branches
set_option maxRecDepth 4096 in
theorem row3733_eq_data : row3733 = Data.row3733 := rfl
def node3733 : Graph := .node 3733 row3733 node3727 node3761
set_option maxRecDepth 4096 in
theorem node3733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3733 :=
  Budget.node_covers _ _ 3733 row3733 node3727 node3761
    node3727_covers node3761_covers (by decide) (by decide) row3733_covers
theorem node3733_eq_data : node3733 = Data.node3733 := by
  have branches : (node3727, node3761) = (Data.node3727, Data.node3761) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3727_eq_data node3761_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3733 row children.1 children.2) row3733_eq_data branches
set_option maxRecDepth 4096 in
theorem row3709_eq_data : row3709 = Data.row3709 := rfl
def node3709 : Graph := .node 3709 row3709 node3691 node3733
set_option maxRecDepth 4096 in
theorem node3709_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3709 :=
  Budget.node_covers _ _ 3709 row3709 node3691 node3733
    node3691_covers node3733_covers (by decide) (by decide) row3709_covers
theorem node3709_eq_data : node3709 = Data.node3709 := by
  have branches : (node3691, node3733) = (Data.node3691, Data.node3733) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3691_eq_data node3733_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3709 row children.1 children.2) row3709_eq_data branches
set_option maxRecDepth 4096 in
theorem row3769_eq_data : row3769 = Data.row3769 := rfl
def node3769 : Graph := .node 3769 row3769 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3769_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3769 :=
  Budget.node_covers _ _ 3769 row3769 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3769_covers
theorem node3769_eq_data : node3769 = Data.node3769 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3769 row children.1 children.2) row3769_eq_data branches
set_option maxRecDepth 4096 in
theorem row3779_eq_data : row3779 = Data.row3779 := rfl
def node3779 : Graph := .node 3779 row3779 node3769 Graph.empty
set_option maxRecDepth 4096 in
theorem node3779_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3779 :=
  Budget.node_covers _ _ 3779 row3779 node3769 Graph.empty
    node3769_covers empty_covers (by decide) (by decide) row3779_covers
theorem node3779_eq_data : node3779 = Data.node3779 := by
  have branches : (node3769, Graph.empty) = (Data.node3769, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3769_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3779 row children.1 children.2) row3779_eq_data branches
set_option maxRecDepth 4096 in
theorem row3797_eq_data : row3797 = Data.row3797 := rfl
def node3797 : Graph := .node 3797 row3797 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3797_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3797 :=
  Budget.node_covers _ _ 3797 row3797 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3797_covers
theorem node3797_eq_data : node3797 = Data.node3797 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3797 row children.1 children.2) row3797_eq_data branches
set_option maxRecDepth 4096 in
theorem row3803_eq_data : row3803 = Data.row3803 := rfl
def node3803 : Graph := .node 3803 row3803 node3797 Graph.empty
set_option maxRecDepth 4096 in
theorem node3803_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3803 :=
  Budget.node_covers _ _ 3803 row3803 node3797 Graph.empty
    node3797_covers empty_covers (by decide) (by decide) row3803_covers
theorem node3803_eq_data : node3803 = Data.node3803 := by
  have branches : (node3797, Graph.empty) = (Data.node3797, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3797_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3803 row children.1 children.2) row3803_eq_data branches
set_option maxRecDepth 4096 in
theorem row3793_eq_data : row3793 = Data.row3793 := rfl
def node3793 : Graph := .node 3793 row3793 node3779 node3803
set_option maxRecDepth 4096 in
theorem node3793_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3793 :=
  Budget.node_covers _ _ 3793 row3793 node3779 node3803
    node3779_covers node3803_covers (by decide) (by decide) row3793_covers
theorem node3793_eq_data : node3793 = Data.node3793 := by
  have branches : (node3779, node3803) = (Data.node3779, Data.node3803) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3779_eq_data node3803_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3793 row children.1 children.2) row3793_eq_data branches
set_option maxRecDepth 4096 in
theorem row3823_eq_data : row3823 = Data.row3823 := rfl
def node3823 : Graph := .node 3823 row3823 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3823_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3823 :=
  Budget.node_covers _ _ 3823 row3823 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3823_covers
theorem node3823_eq_data : node3823 = Data.node3823 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3823 row children.1 children.2) row3823_eq_data branches
set_option maxRecDepth 4096 in
theorem row3833_eq_data : row3833 = Data.row3833 := rfl
def node3833 : Graph := .node 3833 row3833 node3823 Graph.empty
set_option maxRecDepth 4096 in
theorem node3833_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3833 :=
  Budget.node_covers _ _ 3833 row3833 node3823 Graph.empty
    node3823_covers empty_covers (by decide) (by decide) row3833_covers
theorem node3833_eq_data : node3833 = Data.node3833 := by
  have branches : (node3823, Graph.empty) = (Data.node3823, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3823_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3833 row children.1 children.2) row3833_eq_data branches
set_option maxRecDepth 4096 in
theorem row3851_eq_data : row3851 = Data.row3851 := rfl
def node3851 : Graph := .node 3851 row3851 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3851_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3851 :=
  Budget.node_covers _ _ 3851 row3851 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3851_covers
theorem node3851_eq_data : node3851 = Data.node3851 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3851 row children.1 children.2) row3851_eq_data branches
set_option maxRecDepth 4096 in
theorem row3853_eq_data : row3853 = Data.row3853 := rfl
def node3853 : Graph := .node 3853 row3853 node3851 Graph.empty
set_option maxRecDepth 4096 in
theorem node3853_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3853 :=
  Budget.node_covers _ _ 3853 row3853 node3851 Graph.empty
    node3851_covers empty_covers (by decide) (by decide) row3853_covers
theorem node3853_eq_data : node3853 = Data.node3853 := by
  have branches : (node3851, Graph.empty) = (Data.node3851, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3851_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3853 row children.1 children.2) row3853_eq_data branches
set_option maxRecDepth 4096 in
theorem row3847_eq_data : row3847 = Data.row3847 := rfl
def node3847 : Graph := .node 3847 row3847 node3833 node3853
set_option maxRecDepth 4096 in
theorem node3847_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3847 :=
  Budget.node_covers _ _ 3847 row3847 node3833 node3853
    node3833_covers node3853_covers (by decide) (by decide) row3847_covers
theorem node3847_eq_data : node3847 = Data.node3847 := by
  have branches : (node3833, node3853) = (Data.node3833, Data.node3853) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3833_eq_data node3853_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3847 row children.1 children.2) row3847_eq_data branches
set_option maxRecDepth 4096 in
theorem row3821_eq_data : row3821 = Data.row3821 := rfl
def node3821 : Graph := .node 3821 row3821 node3793 node3847
set_option maxRecDepth 4096 in
theorem node3821_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3821 :=
  Budget.node_covers _ _ 3821 row3821 node3793 node3847
    node3793_covers node3847_covers (by decide) (by decide) row3821_covers
theorem node3821_eq_data : node3821 = Data.node3821 := by
  have branches : (node3793, node3847) = (Data.node3793, Data.node3847) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3793_eq_data node3847_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3821 row children.1 children.2) row3821_eq_data branches
set_option maxRecDepth 4096 in
theorem row3767_eq_data : row3767 = Data.row3767 := rfl
def node3767 : Graph := .node 3767 row3767 node3709 node3821
set_option maxRecDepth 4096 in
theorem node3767_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3767 :=
  Budget.node_covers _ _ 3767 row3767 node3709 node3821
    node3709_covers node3821_covers (by decide) (by decide) row3767_covers
theorem node3767_eq_data : node3767 = Data.node3767 := by
  have branches : (node3709, node3821) = (Data.node3709, Data.node3821) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3709_eq_data node3821_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3767 row children.1 children.2) row3767_eq_data branches
set_option maxRecDepth 4096 in
theorem row3671_eq_data : row3671 = Data.row3671 := rfl
def node3671 : Graph := .node 3671 row3671 node3571 node3767
set_option maxRecDepth 4096 in
theorem node3671_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3671 :=
  Budget.node_covers _ _ 3671 row3671 node3571 node3767
    node3571_covers node3767_covers (by decide) (by decide) row3671_covers
theorem node3671_eq_data : node3671 = Data.node3671 := by
  have branches : (node3571, node3767) = (Data.node3571, Data.node3767) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3571_eq_data node3767_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3671 row children.1 children.2) row3671_eq_data branches
set_option maxRecDepth 4096 in
theorem row3877_eq_data : row3877 = Data.row3877 := rfl
def node3877 : Graph := .node 3877 row3877 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3877_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3877 :=
  Budget.node_covers _ _ 3877 row3877 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3877_covers
theorem node3877_eq_data : node3877 = Data.node3877 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3877 row children.1 children.2) row3877_eq_data branches
set_option maxRecDepth 4096 in
theorem row3881_eq_data : row3881 = Data.row3881 := rfl
def node3881 : Graph := .node 3881 row3881 node3877 Graph.empty
set_option maxRecDepth 4096 in
theorem node3881_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3881 :=
  Budget.node_covers _ _ 3881 row3881 node3877 Graph.empty
    node3877_covers empty_covers (by decide) (by decide) row3881_covers
theorem node3881_eq_data : node3881 = Data.node3881 := by
  have branches : (node3877, Graph.empty) = (Data.node3877, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3877_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3881 row children.1 children.2) row3881_eq_data branches
set_option maxRecDepth 4096 in
theorem row3907_eq_data : row3907 = Data.row3907 := rfl
def node3907 : Graph := .node 3907 row3907 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3907 :=
  Budget.node_covers _ _ 3907 row3907 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3907_covers
theorem node3907_eq_data : node3907 = Data.node3907 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3907 row children.1 children.2) row3907_eq_data branches
set_option maxRecDepth 4096 in
theorem row3911_eq_data : row3911 = Data.row3911 := rfl
def node3911 : Graph := .node 3911 row3911 node3907 Graph.empty
set_option maxRecDepth 4096 in
theorem node3911_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3911 :=
  Budget.node_covers _ _ 3911 row3911 node3907 Graph.empty
    node3907_covers empty_covers (by decide) (by decide) row3911_covers
theorem node3911_eq_data : node3911 = Data.node3911 := by
  have branches : (node3907, Graph.empty) = (Data.node3907, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3907_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3911 row children.1 children.2) row3911_eq_data branches
set_option maxRecDepth 4096 in
theorem row3889_eq_data : row3889 = Data.row3889 := rfl
def node3889 : Graph := .node 3889 row3889 node3881 node3911
set_option maxRecDepth 4096 in
theorem node3889_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3889 :=
  Budget.node_covers _ _ 3889 row3889 node3881 node3911
    node3881_covers node3911_covers (by decide) (by decide) row3889_covers
theorem node3889_eq_data : node3889 = Data.node3889 := by
  have branches : (node3881, node3911) = (Data.node3881, Data.node3911) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3881_eq_data node3911_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3889 row children.1 children.2) row3889_eq_data branches
set_option maxRecDepth 4096 in
theorem row3919_eq_data : row3919 = Data.row3919 := rfl
def node3919 : Graph := .node 3919 row3919 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3919_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3919 :=
  Budget.node_covers _ _ 3919 row3919 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3919_covers
theorem node3919_eq_data : node3919 = Data.node3919 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3919 row children.1 children.2) row3919_eq_data branches
set_option maxRecDepth 4096 in
theorem row3923_eq_data : row3923 = Data.row3923 := rfl
def node3923 : Graph := .node 3923 row3923 node3919 Graph.empty
set_option maxRecDepth 4096 in
theorem node3923_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3923 :=
  Budget.node_covers _ _ 3923 row3923 node3919 Graph.empty
    node3919_covers empty_covers (by decide) (by decide) row3923_covers
theorem node3923_eq_data : node3923 = Data.node3923 := by
  have branches : (node3919, Graph.empty) = (Data.node3919, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3919_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3923 row children.1 children.2) row3923_eq_data branches
set_option maxRecDepth 4096 in
theorem row3931_eq_data : row3931 = Data.row3931 := rfl
def node3931 : Graph := .node 3931 row3931 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3931_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3931 :=
  Budget.node_covers _ _ 3931 row3931 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3931_covers
theorem node3931_eq_data : node3931 = Data.node3931 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3931 row children.1 children.2) row3931_eq_data branches
set_option maxRecDepth 4096 in
theorem row3943_eq_data : row3943 = Data.row3943 := rfl
def node3943 : Graph := .node 3943 row3943 node3931 Graph.empty
set_option maxRecDepth 4096 in
theorem node3943_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3943 :=
  Budget.node_covers _ _ 3943 row3943 node3931 Graph.empty
    node3931_covers empty_covers (by decide) (by decide) row3943_covers
theorem node3943_eq_data : node3943 = Data.node3943 := by
  have branches : (node3931, Graph.empty) = (Data.node3931, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3931_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3943 row children.1 children.2) row3943_eq_data branches
set_option maxRecDepth 4096 in
theorem row3929_eq_data : row3929 = Data.row3929 := rfl
def node3929 : Graph := .node 3929 row3929 node3923 node3943
set_option maxRecDepth 4096 in
theorem node3929_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3929 :=
  Budget.node_covers _ _ 3929 row3929 node3923 node3943
    node3923_covers node3943_covers (by decide) (by decide) row3929_covers
theorem node3929_eq_data : node3929 = Data.node3929 := by
  have branches : (node3923, node3943) = (Data.node3923, Data.node3943) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3923_eq_data node3943_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3929 row children.1 children.2) row3929_eq_data branches
set_option maxRecDepth 4096 in
theorem row3917_eq_data : row3917 = Data.row3917 := rfl
def node3917 : Graph := .node 3917 row3917 node3889 node3929
set_option maxRecDepth 4096 in
theorem node3917_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3917 :=
  Budget.node_covers _ _ 3917 row3917 node3889 node3929
    node3889_covers node3929_covers (by decide) (by decide) row3917_covers
theorem node3917_eq_data : node3917 = Data.node3917 := by
  have branches : (node3889, node3929) = (Data.node3889, Data.node3929) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3889_eq_data node3929_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3917 row children.1 children.2) row3917_eq_data branches
set_option maxRecDepth 4096 in
theorem row3967_eq_data : row3967 = Data.row3967 := rfl
def node3967 : Graph := .node 3967 row3967 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3967_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3967 :=
  Budget.node_covers _ _ 3967 row3967 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3967_covers
theorem node3967_eq_data : node3967 = Data.node3967 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3967 row children.1 children.2) row3967_eq_data branches
set_option maxRecDepth 4096 in
theorem row3989_eq_data : row3989 = Data.row3989 := rfl
def node3989 : Graph := .node 3989 row3989 node3967 Graph.empty
set_option maxRecDepth 4096 in
theorem node3989_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3989 :=
  Budget.node_covers _ _ 3989 row3989 node3967 Graph.empty
    node3967_covers empty_covers (by decide) (by decide) row3989_covers
theorem node3989_eq_data : node3989 = Data.node3989 := by
  have branches : (node3967, Graph.empty) = (Data.node3967, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3967_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3989 row children.1 children.2) row3989_eq_data branches
set_option maxRecDepth 4096 in
theorem row4003_eq_data : row4003 = Data.row4003 := rfl
def node4003 : Graph := .node 4003 row4003 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4003_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4003 :=
  Budget.node_covers _ _ 4003 row4003 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4003_covers
theorem node4003_eq_data : node4003 = Data.node4003 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4003 row children.1 children.2) row4003_eq_data branches
set_option maxRecDepth 4096 in
theorem row4007_eq_data : row4007 = Data.row4007 := rfl
def node4007 : Graph := .node 4007 row4007 node4003 Graph.empty
set_option maxRecDepth 4096 in
theorem node4007_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4007 :=
  Budget.node_covers _ _ 4007 row4007 node4003 Graph.empty
    node4003_covers empty_covers (by decide) (by decide) row4007_covers
theorem node4007_eq_data : node4007 = Data.node4007 := by
  have branches : (node4003, Graph.empty) = (Data.node4003, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4003_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4007 row children.1 children.2) row4007_eq_data branches
set_option maxRecDepth 4096 in
theorem row4001_eq_data : row4001 = Data.row4001 := rfl
def node4001 : Graph := .node 4001 row4001 node3989 node4007
set_option maxRecDepth 4096 in
theorem node4001_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4001 :=
  Budget.node_covers _ _ 4001 row4001 node3989 node4007
    node3989_covers node4007_covers (by decide) (by decide) row4001_covers
theorem node4001_eq_data : node4001 = Data.node4001 := by
  have branches : (node3989, node4007) = (Data.node3989, Data.node4007) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3989_eq_data node4007_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4001 row children.1 children.2) row4001_eq_data branches
set_option maxRecDepth 4096 in
theorem row4019_eq_data : row4019 = Data.row4019 := rfl
def node4019 : Graph := .node 4019 row4019 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4019_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4019 :=
  Budget.node_covers _ _ 4019 row4019 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4019_covers
theorem node4019_eq_data : node4019 = Data.node4019 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4019 row children.1 children.2) row4019_eq_data branches
set_option maxRecDepth 4096 in
theorem row4021_eq_data : row4021 = Data.row4021 := rfl
def node4021 : Graph := .node 4021 row4021 node4019 Graph.empty
set_option maxRecDepth 4096 in
theorem node4021_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4021 :=
  Budget.node_covers _ _ 4021 row4021 node4019 Graph.empty
    node4019_covers empty_covers (by decide) (by decide) row4021_covers
theorem node4021_eq_data : node4021 = Data.node4021 := by
  have branches : (node4019, Graph.empty) = (Data.node4019, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4019_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4021 row children.1 children.2) row4021_eq_data branches
set_option maxRecDepth 4096 in
theorem row4049_eq_data : row4049 = Data.row4049 := rfl
def node4049 : Graph := .node 4049 row4049 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4049_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4049 :=
  Budget.node_covers _ _ 4049 row4049 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4049_covers
theorem node4049_eq_data : node4049 = Data.node4049 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4049 row children.1 children.2) row4049_eq_data branches
set_option maxRecDepth 4096 in
theorem row4051_eq_data : row4051 = Data.row4051 := rfl
def node4051 : Graph := .node 4051 row4051 node4049 Graph.empty
set_option maxRecDepth 4096 in
theorem node4051_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4051 :=
  Budget.node_covers _ _ 4051 row4051 node4049 Graph.empty
    node4049_covers empty_covers (by decide) (by decide) row4051_covers
theorem node4051_eq_data : node4051 = Data.node4051 := by
  have branches : (node4049, Graph.empty) = (Data.node4049, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4049_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4051 row children.1 children.2) row4051_eq_data branches
set_option maxRecDepth 4096 in
theorem row4027_eq_data : row4027 = Data.row4027 := rfl
def node4027 : Graph := .node 4027 row4027 node4021 node4051
set_option maxRecDepth 4096 in
theorem node4027_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4027 :=
  Budget.node_covers _ _ 4027 row4027 node4021 node4051
    node4021_covers node4051_covers (by decide) (by decide) row4027_covers
theorem node4027_eq_data : node4027 = Data.node4027 := by
  have branches : (node4021, node4051) = (Data.node4021, Data.node4051) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4021_eq_data node4051_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4027 row children.1 children.2) row4027_eq_data branches
set_option maxRecDepth 4096 in
theorem row4013_eq_data : row4013 = Data.row4013 := rfl
def node4013 : Graph := .node 4013 row4013 node4001 node4027
set_option maxRecDepth 4096 in
theorem node4013_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4013 :=
  Budget.node_covers _ _ 4013 row4013 node4001 node4027
    node4001_covers node4027_covers (by decide) (by decide) row4013_covers
theorem node4013_eq_data : node4013 = Data.node4013 := by
  have branches : (node4001, node4027) = (Data.node4001, Data.node4027) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4001_eq_data node4027_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4013 row children.1 children.2) row4013_eq_data branches
set_option maxRecDepth 4096 in
theorem row3947_eq_data : row3947 = Data.row3947 := rfl
def node3947 : Graph := .node 3947 row3947 node3917 node4013
set_option maxRecDepth 4096 in
theorem node3947_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3947 :=
  Budget.node_covers _ _ 3947 row3947 node3917 node4013
    node3917_covers node4013_covers (by decide) (by decide) row3947_covers
theorem node3947_eq_data : node3947 = Data.node3947 := by
  have branches : (node3917, node4013) = (Data.node3917, Data.node4013) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3917_eq_data node4013_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3947 row children.1 children.2) row3947_eq_data branches
set_option maxRecDepth 4096 in
theorem row4073_eq_data : row4073 = Data.row4073 := rfl
def node4073 : Graph := .node 4073 row4073 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4073_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4073 :=
  Budget.node_covers _ _ 4073 row4073 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4073_covers
theorem node4073_eq_data : node4073 = Data.node4073 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4073 row children.1 children.2) row4073_eq_data branches
set_option maxRecDepth 4096 in
theorem row4079_eq_data : row4079 = Data.row4079 := rfl
def node4079 : Graph := .node 4079 row4079 node4073 Graph.empty
set_option maxRecDepth 4096 in
theorem node4079_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4079 :=
  Budget.node_covers _ _ 4079 row4079 node4073 Graph.empty
    node4073_covers empty_covers (by decide) (by decide) row4079_covers
theorem node4079_eq_data : node4079 = Data.node4079 := by
  have branches : (node4073, Graph.empty) = (Data.node4073, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4073_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4079 row children.1 children.2) row4079_eq_data branches
set_option maxRecDepth 4096 in
theorem row4093_eq_data : row4093 = Data.row4093 := rfl
def node4093 : Graph := .node 4093 row4093 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4093_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4093 :=
  Budget.node_covers _ _ 4093 row4093 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4093_covers
theorem node4093_eq_data : node4093 = Data.node4093 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4093 row children.1 children.2) row4093_eq_data branches
set_option maxRecDepth 4096 in
theorem row4099_eq_data : row4099 = Data.row4099 := rfl
def node4099 : Graph := .node 4099 row4099 node4093 Graph.empty
set_option maxRecDepth 4096 in
theorem node4099_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4099 :=
  Budget.node_covers _ _ 4099 row4099 node4093 Graph.empty
    node4093_covers empty_covers (by decide) (by decide) row4099_covers
theorem node4099_eq_data : node4099 = Data.node4099 := by
  have branches : (node4093, Graph.empty) = (Data.node4093, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4093_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4099 row children.1 children.2) row4099_eq_data branches
set_option maxRecDepth 4096 in
theorem row4091_eq_data : row4091 = Data.row4091 := rfl
def node4091 : Graph := .node 4091 row4091 node4079 node4099
set_option maxRecDepth 4096 in
theorem node4091_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4091 :=
  Budget.node_covers _ _ 4091 row4091 node4079 node4099
    node4079_covers node4099_covers (by decide) (by decide) row4091_covers
theorem node4091_eq_data : node4091 = Data.node4091 := by
  have branches : (node4079, node4099) = (Data.node4079, Data.node4099) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4079_eq_data node4099_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4091 row children.1 children.2) row4091_eq_data branches
set_option maxRecDepth 4096 in
theorem row4127_eq_data : row4127 = Data.row4127 := rfl
def node4127 : Graph := .node 4127 row4127 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4127_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4127 :=
  Budget.node_covers _ _ 4127 row4127 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4127_covers
theorem node4127_eq_data : node4127 = Data.node4127 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4127 row children.1 children.2) row4127_eq_data branches
set_option maxRecDepth 4096 in
theorem row4129_eq_data : row4129 = Data.row4129 := rfl
def node4129 : Graph := .node 4129 row4129 node4127 Graph.empty
set_option maxRecDepth 4096 in
theorem node4129_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4129 :=
  Budget.node_covers _ _ 4129 row4129 node4127 Graph.empty
    node4127_covers empty_covers (by decide) (by decide) row4129_covers
theorem node4129_eq_data : node4129 = Data.node4129 := by
  have branches : (node4127, Graph.empty) = (Data.node4127, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4127_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4129 row children.1 children.2) row4129_eq_data branches
set_option maxRecDepth 4096 in
theorem row4139_eq_data : row4139 = Data.row4139 := rfl
def node4139 : Graph := .node 4139 row4139 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4139_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4139 :=
  Budget.node_covers _ _ 4139 row4139 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4139_covers
theorem node4139_eq_data : node4139 = Data.node4139 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4139 row children.1 children.2) row4139_eq_data branches
set_option maxRecDepth 4096 in
theorem row4153_eq_data : row4153 = Data.row4153 := rfl
def node4153 : Graph := .node 4153 row4153 node4139 Graph.empty
set_option maxRecDepth 4096 in
theorem node4153_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4153 :=
  Budget.node_covers _ _ 4153 row4153 node4139 Graph.empty
    node4139_covers empty_covers (by decide) (by decide) row4153_covers
theorem node4153_eq_data : node4153 = Data.node4153 := by
  have branches : (node4139, Graph.empty) = (Data.node4139, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4139_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4153 row children.1 children.2) row4153_eq_data branches
set_option maxRecDepth 4096 in
theorem row4133_eq_data : row4133 = Data.row4133 := rfl
def node4133 : Graph := .node 4133 row4133 node4129 node4153
set_option maxRecDepth 4096 in
theorem node4133_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4133 :=
  Budget.node_covers _ _ 4133 row4133 node4129 node4153
    node4129_covers node4153_covers (by decide) (by decide) row4133_covers
theorem node4133_eq_data : node4133 = Data.node4133 := by
  have branches : (node4129, node4153) = (Data.node4129, Data.node4153) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4129_eq_data node4153_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4133 row children.1 children.2) row4133_eq_data branches
set_option maxRecDepth 4096 in
theorem row4111_eq_data : row4111 = Data.row4111 := rfl
def node4111 : Graph := .node 4111 row4111 node4091 node4133
set_option maxRecDepth 4096 in
theorem node4111_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4111 :=
  Budget.node_covers _ _ 4111 row4111 node4091 node4133
    node4091_covers node4133_covers (by decide) (by decide) row4111_covers
theorem node4111_eq_data : node4111 = Data.node4111 := by
  have branches : (node4091, node4133) = (Data.node4091, Data.node4133) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4091_eq_data node4133_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4111 row children.1 children.2) row4111_eq_data branches
set_option maxRecDepth 4096 in
theorem row4159_eq_data : row4159 = Data.row4159 := rfl
def node4159 : Graph := .node 4159 row4159 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4159_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4159 :=
  Budget.node_covers _ _ 4159 row4159 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4159_covers
theorem node4159_eq_data : node4159 = Data.node4159 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4159 row children.1 children.2) row4159_eq_data branches
set_option maxRecDepth 4096 in
theorem row4177_eq_data : row4177 = Data.row4177 := rfl
def node4177 : Graph := .node 4177 row4177 node4159 Graph.empty
set_option maxRecDepth 4096 in
theorem node4177_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4177 :=
  Budget.node_covers _ _ 4177 row4177 node4159 Graph.empty
    node4159_covers empty_covers (by decide) (by decide) row4177_covers
theorem node4177_eq_data : node4177 = Data.node4177 := by
  have branches : (node4159, Graph.empty) = (Data.node4159, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4159_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4177 row children.1 children.2) row4177_eq_data branches
set_option maxRecDepth 4096 in
theorem row4211_eq_data : row4211 = Data.row4211 := rfl
def node4211 : Graph := .node 4211 row4211 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4211_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4211 :=
  Budget.node_covers _ _ 4211 row4211 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4211_covers
theorem node4211_eq_data : node4211 = Data.node4211 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4211 row children.1 children.2) row4211_eq_data branches
set_option maxRecDepth 4096 in
theorem row4217_eq_data : row4217 = Data.row4217 := rfl
def node4217 : Graph := .node 4217 row4217 node4211 Graph.empty
set_option maxRecDepth 4096 in
theorem node4217_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4217 :=
  Budget.node_covers _ _ 4217 row4217 node4211 Graph.empty
    node4211_covers empty_covers (by decide) (by decide) row4217_covers
theorem node4217_eq_data : node4217 = Data.node4217 := by
  have branches : (node4211, Graph.empty) = (Data.node4211, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4211_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4217 row children.1 children.2) row4217_eq_data branches
set_option maxRecDepth 4096 in
theorem row4201_eq_data : row4201 = Data.row4201 := rfl
def node4201 : Graph := .node 4201 row4201 node4177 node4217
set_option maxRecDepth 4096 in
theorem node4201_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4201 :=
  Budget.node_covers _ _ 4201 row4201 node4177 node4217
    node4177_covers node4217_covers (by decide) (by decide) row4201_covers
theorem node4201_eq_data : node4201 = Data.node4201 := by
  have branches : (node4177, node4217) = (Data.node4177, Data.node4217) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4177_eq_data node4217_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4201 row children.1 children.2) row4201_eq_data branches
set_option maxRecDepth 4096 in
theorem row4229_eq_data : row4229 = Data.row4229 := rfl
def node4229 : Graph := .node 4229 row4229 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4229_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4229 :=
  Budget.node_covers _ _ 4229 row4229 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4229_covers
theorem node4229_eq_data : node4229 = Data.node4229 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4229 row children.1 children.2) row4229_eq_data branches
set_option maxRecDepth 4096 in
theorem row4231_eq_data : row4231 = Data.row4231 := rfl
def node4231 : Graph := .node 4231 row4231 node4229 Graph.empty
set_option maxRecDepth 4096 in
theorem node4231_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4231 :=
  Budget.node_covers _ _ 4231 row4231 node4229 Graph.empty
    node4229_covers empty_covers (by decide) (by decide) row4231_covers
theorem node4231_eq_data : node4231 = Data.node4231 := by
  have branches : (node4229, Graph.empty) = (Data.node4229, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4229_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4231 row children.1 children.2) row4231_eq_data branches
set_option maxRecDepth 4096 in
theorem row4243_eq_data : row4243 = Data.row4243 := rfl
def node4243 : Graph := .node 4243 row4243 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4243_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4243 :=
  Budget.node_covers _ _ 4243 row4243 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4243_covers
theorem node4243_eq_data : node4243 = Data.node4243 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4243 row children.1 children.2) row4243_eq_data branches
set_option maxRecDepth 4096 in
theorem row4253_eq_data : row4253 = Data.row4253 := rfl
def node4253 : Graph := .node 4253 row4253 node4243 Graph.empty
set_option maxRecDepth 4096 in
theorem node4253_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4253 :=
  Budget.node_covers _ _ 4253 row4253 node4243 Graph.empty
    node4243_covers empty_covers (by decide) (by decide) row4253_covers
theorem node4253_eq_data : node4253 = Data.node4253 := by
  have branches : (node4243, Graph.empty) = (Data.node4243, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4243_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4253 row children.1 children.2) row4253_eq_data branches
set_option maxRecDepth 4096 in
theorem row4241_eq_data : row4241 = Data.row4241 := rfl
def node4241 : Graph := .node 4241 row4241 node4231 node4253
set_option maxRecDepth 4096 in
theorem node4241_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4241 :=
  Budget.node_covers _ _ 4241 row4241 node4231 node4253
    node4231_covers node4253_covers (by decide) (by decide) row4241_covers
theorem node4241_eq_data : node4241 = Data.node4241 := by
  have branches : (node4231, node4253) = (Data.node4231, Data.node4253) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4231_eq_data node4253_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4241 row children.1 children.2) row4241_eq_data branches
set_option maxRecDepth 4096 in
theorem row4219_eq_data : row4219 = Data.row4219 := rfl
def node4219 : Graph := .node 4219 row4219 node4201 node4241
set_option maxRecDepth 4096 in
theorem node4219_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4219 :=
  Budget.node_covers _ _ 4219 row4219 node4201 node4241
    node4201_covers node4241_covers (by decide) (by decide) row4219_covers
theorem node4219_eq_data : node4219 = Data.node4219 := by
  have branches : (node4201, node4241) = (Data.node4201, Data.node4241) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4201_eq_data node4241_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4219 row children.1 children.2) row4219_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
