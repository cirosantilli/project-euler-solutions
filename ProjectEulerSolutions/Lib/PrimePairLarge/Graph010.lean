import ProjectEulerSolutions.Lib.PrimePairLarge.Graph009
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row4801_eq_data : row4801 = Data.row4801 := rfl
def node4801 : Graph := .node 4801 row4801 node4799 Graph.empty
set_option maxRecDepth 4096 in
theorem node4801_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4801 :=
  Budget.node_covers _ _ 4801 row4801 node4799 Graph.empty
    node4799_covers empty_covers (by decide) (by decide) row4801_covers
theorem node4801_eq_data : node4801 = Data.node4801 := by
  have branches : (node4799, Graph.empty) = (Data.node4799, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4799_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4801 row children.1 children.2) row4801_eq_data branches
set_option maxRecDepth 4096 in
theorem row4817_eq_data : row4817 = Data.row4817 := rfl
def node4817 : Graph := .node 4817 row4817 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4817_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4817 :=
  Budget.node_covers _ _ 4817 row4817 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4817_covers
theorem node4817_eq_data : node4817 = Data.node4817 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4817 row children.1 children.2) row4817_eq_data branches
set_option maxRecDepth 4096 in
theorem row4831_eq_data : row4831 = Data.row4831 := rfl
def node4831 : Graph := .node 4831 row4831 node4817 Graph.empty
set_option maxRecDepth 4096 in
theorem node4831_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4831 :=
  Budget.node_covers _ _ 4831 row4831 node4817 Graph.empty
    node4817_covers empty_covers (by decide) (by decide) row4831_covers
theorem node4831_eq_data : node4831 = Data.node4831 := by
  have branches : (node4817, Graph.empty) = (Data.node4817, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4817_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4831 row children.1 children.2) row4831_eq_data branches
set_option maxRecDepth 4096 in
theorem row4813_eq_data : row4813 = Data.row4813 := rfl
def node4813 : Graph := .node 4813 row4813 node4801 node4831
set_option maxRecDepth 4096 in
theorem node4813_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4813 :=
  Budget.node_covers _ _ 4813 row4813 node4801 node4831
    node4801_covers node4831_covers (by decide) (by decide) row4813_covers
theorem node4813_eq_data : node4813 = Data.node4813 := by
  have branches : (node4801, node4831) = (Data.node4801, Data.node4831) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4801_eq_data node4831_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4813 row children.1 children.2) row4813_eq_data branches
set_option maxRecDepth 4096 in
theorem row4871_eq_data : row4871 = Data.row4871 := rfl
def node4871 : Graph := .node 4871 row4871 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4871_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4871 :=
  Budget.node_covers _ _ 4871 row4871 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4871_covers
theorem node4871_eq_data : node4871 = Data.node4871 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4871 row children.1 children.2) row4871_eq_data branches
set_option maxRecDepth 4096 in
theorem row4877_eq_data : row4877 = Data.row4877 := rfl
def node4877 : Graph := .node 4877 row4877 node4871 Graph.empty
set_option maxRecDepth 4096 in
theorem node4877_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4877 :=
  Budget.node_covers _ _ 4877 row4877 node4871 Graph.empty
    node4871_covers empty_covers (by decide) (by decide) row4877_covers
theorem node4877_eq_data : node4877 = Data.node4877 := by
  have branches : (node4871, Graph.empty) = (Data.node4871, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4871_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4877 row children.1 children.2) row4877_eq_data branches
set_option maxRecDepth 4096 in
theorem row4903_eq_data : row4903 = Data.row4903 := rfl
def node4903 : Graph := .node 4903 row4903 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4903_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4903 :=
  Budget.node_covers _ _ 4903 row4903 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4903_covers
theorem node4903_eq_data : node4903 = Data.node4903 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4903 row children.1 children.2) row4903_eq_data branches
set_option maxRecDepth 4096 in
theorem row4909_eq_data : row4909 = Data.row4909 := rfl
def node4909 : Graph := .node 4909 row4909 node4903 Graph.empty
set_option maxRecDepth 4096 in
theorem node4909_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4909 :=
  Budget.node_covers _ _ 4909 row4909 node4903 Graph.empty
    node4903_covers empty_covers (by decide) (by decide) row4909_covers
theorem node4909_eq_data : node4909 = Data.node4909 := by
  have branches : (node4903, Graph.empty) = (Data.node4903, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4903_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4909 row children.1 children.2) row4909_eq_data branches
set_option maxRecDepth 4096 in
theorem row4889_eq_data : row4889 = Data.row4889 := rfl
def node4889 : Graph := .node 4889 row4889 node4877 node4909
set_option maxRecDepth 4096 in
theorem node4889_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4889 :=
  Budget.node_covers _ _ 4889 row4889 node4877 node4909
    node4877_covers node4909_covers (by decide) (by decide) row4889_covers
theorem node4889_eq_data : node4889 = Data.node4889 := by
  have branches : (node4877, node4909) = (Data.node4877, Data.node4909) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4877_eq_data node4909_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4889 row children.1 children.2) row4889_eq_data branches
set_option maxRecDepth 4096 in
theorem row4861_eq_data : row4861 = Data.row4861 := rfl
def node4861 : Graph := .node 4861 row4861 node4813 node4889
set_option maxRecDepth 4096 in
theorem node4861_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4861 :=
  Budget.node_covers _ _ 4861 row4861 node4813 node4889
    node4813_covers node4889_covers (by decide) (by decide) row4861_covers
theorem node4861_eq_data : node4861 = Data.node4861 := by
  have branches : (node4813, node4889) = (Data.node4813, Data.node4889) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4813_eq_data node4889_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4861 row children.1 children.2) row4861_eq_data branches
set_option maxRecDepth 4096 in
theorem row4793_eq_data : row4793 = Data.row4793 := rfl
def node4793 : Graph := .node 4793 row4793 node4733 node4861
set_option maxRecDepth 4096 in
theorem node4793_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4793 :=
  Budget.node_covers _ _ 4793 row4793 node4733 node4861
    node4733_covers node4861_covers (by decide) (by decide) row4793_covers
theorem node4793_eq_data : node4793 = Data.node4793 := by
  have branches : (node4733, node4861) = (Data.node4733, Data.node4861) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4733_eq_data node4861_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4793 row children.1 children.2) row4793_eq_data branches
set_option maxRecDepth 4096 in
theorem row4931_eq_data : row4931 = Data.row4931 := rfl
def node4931 : Graph := .node 4931 row4931 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4931_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4931 :=
  Budget.node_covers _ _ 4931 row4931 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4931_covers
theorem node4931_eq_data : node4931 = Data.node4931 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4931 row children.1 children.2) row4931_eq_data branches
set_option maxRecDepth 4096 in
theorem row4933_eq_data : row4933 = Data.row4933 := rfl
def node4933 : Graph := .node 4933 row4933 node4931 Graph.empty
set_option maxRecDepth 4096 in
theorem node4933_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4933 :=
  Budget.node_covers _ _ 4933 row4933 node4931 Graph.empty
    node4931_covers empty_covers (by decide) (by decide) row4933_covers
theorem node4933_eq_data : node4933 = Data.node4933 := by
  have branches : (node4931, Graph.empty) = (Data.node4931, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4931_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4933 row children.1 children.2) row4933_eq_data branches
set_option maxRecDepth 4096 in
theorem row4943_eq_data : row4943 = Data.row4943 := rfl
def node4943 : Graph := .node 4943 row4943 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4943_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4943 :=
  Budget.node_covers _ _ 4943 row4943 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4943_covers
theorem node4943_eq_data : node4943 = Data.node4943 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4943 row children.1 children.2) row4943_eq_data branches
set_option maxRecDepth 4096 in
theorem row4951_eq_data : row4951 = Data.row4951 := rfl
def node4951 : Graph := .node 4951 row4951 node4943 Graph.empty
set_option maxRecDepth 4096 in
theorem node4951_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4951 :=
  Budget.node_covers _ _ 4951 row4951 node4943 Graph.empty
    node4943_covers empty_covers (by decide) (by decide) row4951_covers
theorem node4951_eq_data : node4951 = Data.node4951 := by
  have branches : (node4943, Graph.empty) = (Data.node4943, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4943_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4951 row children.1 children.2) row4951_eq_data branches
set_option maxRecDepth 4096 in
theorem row4937_eq_data : row4937 = Data.row4937 := rfl
def node4937 : Graph := .node 4937 row4937 node4933 node4951
set_option maxRecDepth 4096 in
theorem node4937_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4937 :=
  Budget.node_covers _ _ 4937 row4937 node4933 node4951
    node4933_covers node4951_covers (by decide) (by decide) row4937_covers
theorem node4937_eq_data : node4937 = Data.node4937 := by
  have branches : (node4933, node4951) = (Data.node4933, Data.node4951) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4933_eq_data node4951_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4937 row children.1 children.2) row4937_eq_data branches
set_option maxRecDepth 4096 in
theorem row4967_eq_data : row4967 = Data.row4967 := rfl
def node4967 : Graph := .node 4967 row4967 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4967_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4967 :=
  Budget.node_covers _ _ 4967 row4967 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4967_covers
theorem node4967_eq_data : node4967 = Data.node4967 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4967 row children.1 children.2) row4967_eq_data branches
set_option maxRecDepth 4096 in
theorem row4969_eq_data : row4969 = Data.row4969 := rfl
def node4969 : Graph := .node 4969 row4969 node4967 Graph.empty
set_option maxRecDepth 4096 in
theorem node4969_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4969 :=
  Budget.node_covers _ _ 4969 row4969 node4967 Graph.empty
    node4967_covers empty_covers (by decide) (by decide) row4969_covers
theorem node4969_eq_data : node4969 = Data.node4969 := by
  have branches : (node4967, Graph.empty) = (Data.node4967, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4967_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4969 row children.1 children.2) row4969_eq_data branches
set_option maxRecDepth 4096 in
theorem row4987_eq_data : row4987 = Data.row4987 := rfl
def node4987 : Graph := .node 4987 row4987 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4987_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4987 :=
  Budget.node_covers _ _ 4987 row4987 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4987_covers
theorem node4987_eq_data : node4987 = Data.node4987 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4987 row children.1 children.2) row4987_eq_data branches
set_option maxRecDepth 4096 in
theorem row4993_eq_data : row4993 = Data.row4993 := rfl
def node4993 : Graph := .node 4993 row4993 node4987 Graph.empty
set_option maxRecDepth 4096 in
theorem node4993_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4993 :=
  Budget.node_covers _ _ 4993 row4993 node4987 Graph.empty
    node4987_covers empty_covers (by decide) (by decide) row4993_covers
theorem node4993_eq_data : node4993 = Data.node4993 := by
  have branches : (node4987, Graph.empty) = (Data.node4987, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4987_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4993 row children.1 children.2) row4993_eq_data branches
set_option maxRecDepth 4096 in
theorem row4973_eq_data : row4973 = Data.row4973 := rfl
def node4973 : Graph := .node 4973 row4973 node4969 node4993
set_option maxRecDepth 4096 in
theorem node4973_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4973 :=
  Budget.node_covers _ _ 4973 row4973 node4969 node4993
    node4969_covers node4993_covers (by decide) (by decide) row4973_covers
theorem node4973_eq_data : node4973 = Data.node4973 := by
  have branches : (node4969, node4993) = (Data.node4969, Data.node4993) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4969_eq_data node4993_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4973 row children.1 children.2) row4973_eq_data branches
set_option maxRecDepth 4096 in
theorem row4957_eq_data : row4957 = Data.row4957 := rfl
def node4957 : Graph := .node 4957 row4957 node4937 node4973
set_option maxRecDepth 4096 in
theorem node4957_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4957 :=
  Budget.node_covers _ _ 4957 row4957 node4937 node4973
    node4937_covers node4973_covers (by decide) (by decide) row4957_covers
theorem node4957_eq_data : node4957 = Data.node4957 := by
  have branches : (node4937, node4973) = (Data.node4937, Data.node4973) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4937_eq_data node4973_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4957 row children.1 children.2) row4957_eq_data branches
set_option maxRecDepth 4096 in
theorem row5003_eq_data : row5003 = Data.row5003 := rfl
def node5003 : Graph := .node 5003 row5003 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5003_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5003 :=
  Budget.node_covers _ _ 5003 row5003 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5003_covers
theorem node5003_eq_data : node5003 = Data.node5003 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5003 row children.1 children.2) row5003_eq_data branches
set_option maxRecDepth 4096 in
theorem row5009_eq_data : row5009 = Data.row5009 := rfl
def node5009 : Graph := .node 5009 row5009 node5003 Graph.empty
set_option maxRecDepth 4096 in
theorem node5009_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5009 :=
  Budget.node_covers _ _ 5009 row5009 node5003 Graph.empty
    node5003_covers empty_covers (by decide) (by decide) row5009_covers
theorem node5009_eq_data : node5009 = Data.node5009 := by
  have branches : (node5003, Graph.empty) = (Data.node5003, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5003_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5009 row children.1 children.2) row5009_eq_data branches
set_option maxRecDepth 4096 in
theorem row5021_eq_data : row5021 = Data.row5021 := rfl
def node5021 : Graph := .node 5021 row5021 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5021_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5021 :=
  Budget.node_covers _ _ 5021 row5021 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5021_covers
theorem node5021_eq_data : node5021 = Data.node5021 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5021 row children.1 children.2) row5021_eq_data branches
set_option maxRecDepth 4096 in
theorem row5023_eq_data : row5023 = Data.row5023 := rfl
def node5023 : Graph := .node 5023 row5023 node5021 Graph.empty
set_option maxRecDepth 4096 in
theorem node5023_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5023 :=
  Budget.node_covers _ _ 5023 row5023 node5021 Graph.empty
    node5021_covers empty_covers (by decide) (by decide) row5023_covers
theorem node5023_eq_data : node5023 = Data.node5023 := by
  have branches : (node5021, Graph.empty) = (Data.node5021, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5021_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5023 row children.1 children.2) row5023_eq_data branches
set_option maxRecDepth 4096 in
theorem row5011_eq_data : row5011 = Data.row5011 := rfl
def node5011 : Graph := .node 5011 row5011 node5009 node5023
set_option maxRecDepth 4096 in
theorem node5011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5011 :=
  Budget.node_covers _ _ 5011 row5011 node5009 node5023
    node5009_covers node5023_covers (by decide) (by decide) row5011_covers
theorem node5011_eq_data : node5011 = Data.node5011 := by
  have branches : (node5009, node5023) = (Data.node5009, Data.node5023) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5009_eq_data node5023_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5011 row children.1 children.2) row5011_eq_data branches
set_option maxRecDepth 4096 in
theorem row5051_eq_data : row5051 = Data.row5051 := rfl
def node5051 : Graph := .node 5051 row5051 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5051_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5051 :=
  Budget.node_covers _ _ 5051 row5051 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5051_covers
theorem node5051_eq_data : node5051 = Data.node5051 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5051 row children.1 children.2) row5051_eq_data branches
set_option maxRecDepth 4096 in
theorem row5059_eq_data : row5059 = Data.row5059 := rfl
def node5059 : Graph := .node 5059 row5059 node5051 Graph.empty
set_option maxRecDepth 4096 in
theorem node5059_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5059 :=
  Budget.node_covers _ _ 5059 row5059 node5051 Graph.empty
    node5051_covers empty_covers (by decide) (by decide) row5059_covers
theorem node5059_eq_data : node5059 = Data.node5059 := by
  have branches : (node5051, Graph.empty) = (Data.node5051, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5051_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5059 row children.1 children.2) row5059_eq_data branches
set_option maxRecDepth 4096 in
theorem row5081_eq_data : row5081 = Data.row5081 := rfl
def node5081 : Graph := .node 5081 row5081 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5081_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5081 :=
  Budget.node_covers _ _ 5081 row5081 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5081_covers
theorem node5081_eq_data : node5081 = Data.node5081 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5081 row children.1 children.2) row5081_eq_data branches
set_option maxRecDepth 4096 in
theorem row5087_eq_data : row5087 = Data.row5087 := rfl
def node5087 : Graph := .node 5087 row5087 node5081 Graph.empty
set_option maxRecDepth 4096 in
theorem node5087_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5087 :=
  Budget.node_covers _ _ 5087 row5087 node5081 Graph.empty
    node5081_covers empty_covers (by decide) (by decide) row5087_covers
theorem node5087_eq_data : node5087 = Data.node5087 := by
  have branches : (node5081, Graph.empty) = (Data.node5081, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5081_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5087 row children.1 children.2) row5087_eq_data branches
set_option maxRecDepth 4096 in
theorem row5077_eq_data : row5077 = Data.row5077 := rfl
def node5077 : Graph := .node 5077 row5077 node5059 node5087
set_option maxRecDepth 4096 in
theorem node5077_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5077 :=
  Budget.node_covers _ _ 5077 row5077 node5059 node5087
    node5059_covers node5087_covers (by decide) (by decide) row5077_covers
theorem node5077_eq_data : node5077 = Data.node5077 := by
  have branches : (node5059, node5087) = (Data.node5059, Data.node5087) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5059_eq_data node5087_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5077 row children.1 children.2) row5077_eq_data branches
set_option maxRecDepth 4096 in
theorem row5039_eq_data : row5039 = Data.row5039 := rfl
def node5039 : Graph := .node 5039 row5039 node5011 node5077
set_option maxRecDepth 4096 in
theorem node5039_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5039 :=
  Budget.node_covers _ _ 5039 row5039 node5011 node5077
    node5011_covers node5077_covers (by decide) (by decide) row5039_covers
theorem node5039_eq_data : node5039 = Data.node5039 := by
  have branches : (node5011, node5077) = (Data.node5011, Data.node5077) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5011_eq_data node5077_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5039 row children.1 children.2) row5039_eq_data branches
set_option maxRecDepth 4096 in
theorem row4999_eq_data : row4999 = Data.row4999 := rfl
def node4999 : Graph := .node 4999 row4999 node4957 node5039
set_option maxRecDepth 4096 in
theorem node4999_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4999 :=
  Budget.node_covers _ _ 4999 row4999 node4957 node5039
    node4957_covers node5039_covers (by decide) (by decide) row4999_covers
theorem node4999_eq_data : node4999 = Data.node4999 := by
  have branches : (node4957, node5039) = (Data.node4957, Data.node5039) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4957_eq_data node5039_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4999 row children.1 children.2) row4999_eq_data branches
set_option maxRecDepth 4096 in
theorem row4919_eq_data : row4919 = Data.row4919 := rfl
def node4919 : Graph := .node 4919 row4919 node4793 node4999
set_option maxRecDepth 4096 in
theorem node4919_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4919 :=
  Budget.node_covers _ _ 4919 row4919 node4793 node4999
    node4793_covers node4999_covers (by decide) (by decide) row4919_covers
theorem node4919_eq_data : node4919 = Data.node4919 := by
  have branches : (node4793, node4999) = (Data.node4793, Data.node4999) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4793_eq_data node4999_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4919 row children.1 children.2) row4919_eq_data branches
set_option maxRecDepth 4096 in
theorem row4679_eq_data : row4679 = Data.row4679 := rfl
def node4679 : Graph := .node 4679 row4679 node4483 node4919
set_option maxRecDepth 4096 in
theorem node4679_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4679 :=
  Budget.node_covers _ _ 4679 row4679 node4483 node4919
    node4483_covers node4919_covers (by decide) (by decide) row4679_covers
theorem node4679_eq_data : node4679 = Data.node4679 := by
  have branches : (node4483, node4919) = (Data.node4483, Data.node4919) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4483_eq_data node4919_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4679 row children.1 children.2) row4679_eq_data branches
set_option maxRecDepth 4096 in
theorem row5101_eq_data : row5101 = Data.row5101 := rfl
def node5101 : Graph := .node 5101 row5101 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5101_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5101 :=
  Budget.node_covers _ _ 5101 row5101 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5101_covers
theorem node5101_eq_data : node5101 = Data.node5101 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5101 row children.1 children.2) row5101_eq_data branches
set_option maxRecDepth 4096 in
theorem row5107_eq_data : row5107 = Data.row5107 := rfl
def node5107 : Graph := .node 5107 row5107 node5101 Graph.empty
set_option maxRecDepth 4096 in
theorem node5107_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5107 :=
  Budget.node_covers _ _ 5107 row5107 node5101 Graph.empty
    node5101_covers empty_covers (by decide) (by decide) row5107_covers
theorem node5107_eq_data : node5107 = Data.node5107 := by
  have branches : (node5101, Graph.empty) = (Data.node5101, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5101_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5107 row children.1 children.2) row5107_eq_data branches
set_option maxRecDepth 4096 in
theorem row5119_eq_data : row5119 = Data.row5119 := rfl
def node5119 : Graph := .node 5119 row5119 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5119_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5119 :=
  Budget.node_covers _ _ 5119 row5119 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5119_covers
theorem node5119_eq_data : node5119 = Data.node5119 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5119 row children.1 children.2) row5119_eq_data branches
set_option maxRecDepth 4096 in
theorem row5147_eq_data : row5147 = Data.row5147 := rfl
def node5147 : Graph := .node 5147 row5147 node5119 Graph.empty
set_option maxRecDepth 4096 in
theorem node5147_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5147 :=
  Budget.node_covers _ _ 5147 row5147 node5119 Graph.empty
    node5119_covers empty_covers (by decide) (by decide) row5147_covers
theorem node5147_eq_data : node5147 = Data.node5147 := by
  have branches : (node5119, Graph.empty) = (Data.node5119, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5119_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5147 row children.1 children.2) row5147_eq_data branches
set_option maxRecDepth 4096 in
theorem row5113_eq_data : row5113 = Data.row5113 := rfl
def node5113 : Graph := .node 5113 row5113 node5107 node5147
set_option maxRecDepth 4096 in
theorem node5113_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5113 :=
  Budget.node_covers _ _ 5113 row5113 node5107 node5147
    node5107_covers node5147_covers (by decide) (by decide) row5113_covers
theorem node5113_eq_data : node5113 = Data.node5113 := by
  have branches : (node5107, node5147) = (Data.node5107, Data.node5147) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5107_eq_data node5147_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5113 row children.1 children.2) row5113_eq_data branches
set_option maxRecDepth 4096 in
theorem row5167_eq_data : row5167 = Data.row5167 := rfl
def node5167 : Graph := .node 5167 row5167 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5167_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5167 :=
  Budget.node_covers _ _ 5167 row5167 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5167_covers
theorem node5167_eq_data : node5167 = Data.node5167 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5167 row children.1 children.2) row5167_eq_data branches
set_option maxRecDepth 4096 in
theorem row5171_eq_data : row5171 = Data.row5171 := rfl
def node5171 : Graph := .node 5171 row5171 node5167 Graph.empty
set_option maxRecDepth 4096 in
theorem node5171_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5171 :=
  Budget.node_covers _ _ 5171 row5171 node5167 Graph.empty
    node5167_covers empty_covers (by decide) (by decide) row5171_covers
theorem node5171_eq_data : node5171 = Data.node5171 := by
  have branches : (node5167, Graph.empty) = (Data.node5167, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5167_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5171 row children.1 children.2) row5171_eq_data branches
set_option maxRecDepth 4096 in
theorem row5189_eq_data : row5189 = Data.row5189 := rfl
def node5189 : Graph := .node 5189 row5189 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5189_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5189 :=
  Budget.node_covers _ _ 5189 row5189 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5189_covers
theorem node5189_eq_data : node5189 = Data.node5189 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5189 row children.1 children.2) row5189_eq_data branches
set_option maxRecDepth 4096 in
theorem row5197_eq_data : row5197 = Data.row5197 := rfl
def node5197 : Graph := .node 5197 row5197 node5189 Graph.empty
set_option maxRecDepth 4096 in
theorem node5197_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5197 :=
  Budget.node_covers _ _ 5197 row5197 node5189 Graph.empty
    node5189_covers empty_covers (by decide) (by decide) row5197_covers
theorem node5197_eq_data : node5197 = Data.node5197 := by
  have branches : (node5189, Graph.empty) = (Data.node5189, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5189_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5197 row children.1 children.2) row5197_eq_data branches
set_option maxRecDepth 4096 in
theorem row5179_eq_data : row5179 = Data.row5179 := rfl
def node5179 : Graph := .node 5179 row5179 node5171 node5197
set_option maxRecDepth 4096 in
theorem node5179_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5179 :=
  Budget.node_covers _ _ 5179 row5179 node5171 node5197
    node5171_covers node5197_covers (by decide) (by decide) row5179_covers
theorem node5179_eq_data : node5179 = Data.node5179 := by
  have branches : (node5171, node5197) = (Data.node5171, Data.node5197) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5171_eq_data node5197_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5179 row children.1 children.2) row5179_eq_data branches
set_option maxRecDepth 4096 in
theorem row5153_eq_data : row5153 = Data.row5153 := rfl
def node5153 : Graph := .node 5153 row5153 node5113 node5179
set_option maxRecDepth 4096 in
theorem node5153_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5153 :=
  Budget.node_covers _ _ 5153 row5153 node5113 node5179
    node5113_covers node5179_covers (by decide) (by decide) row5153_covers
theorem node5153_eq_data : node5153 = Data.node5153 := by
  have branches : (node5113, node5179) = (Data.node5113, Data.node5179) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5113_eq_data node5179_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5153 row children.1 children.2) row5153_eq_data branches
set_option maxRecDepth 4096 in
theorem row5227_eq_data : row5227 = Data.row5227 := rfl
def node5227 : Graph := .node 5227 row5227 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5227_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5227 :=
  Budget.node_covers _ _ 5227 row5227 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5227_covers
theorem node5227_eq_data : node5227 = Data.node5227 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5227 row children.1 children.2) row5227_eq_data branches
set_option maxRecDepth 4096 in
theorem row5231_eq_data : row5231 = Data.row5231 := rfl
def node5231 : Graph := .node 5231 row5231 node5227 Graph.empty
set_option maxRecDepth 4096 in
theorem node5231_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5231 :=
  Budget.node_covers _ _ 5231 row5231 node5227 Graph.empty
    node5227_covers empty_covers (by decide) (by decide) row5231_covers
theorem node5231_eq_data : node5231 = Data.node5231 := by
  have branches : (node5227, Graph.empty) = (Data.node5227, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5227_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5231 row children.1 children.2) row5231_eq_data branches
set_option maxRecDepth 4096 in
theorem row5237_eq_data : row5237 = Data.row5237 := rfl
def node5237 : Graph := .node 5237 row5237 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5237_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5237 :=
  Budget.node_covers _ _ 5237 row5237 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5237_covers
theorem node5237_eq_data : node5237 = Data.node5237 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5237 row children.1 children.2) row5237_eq_data branches
set_option maxRecDepth 4096 in
theorem row5261_eq_data : row5261 = Data.row5261 := rfl
def node5261 : Graph := .node 5261 row5261 node5237 Graph.empty
set_option maxRecDepth 4096 in
theorem node5261_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5261 :=
  Budget.node_covers _ _ 5261 row5261 node5237 Graph.empty
    node5237_covers empty_covers (by decide) (by decide) row5261_covers
theorem node5261_eq_data : node5261 = Data.node5261 := by
  have branches : (node5237, Graph.empty) = (Data.node5237, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5237_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5261 row children.1 children.2) row5261_eq_data branches
set_option maxRecDepth 4096 in
theorem row5233_eq_data : row5233 = Data.row5233 := rfl
def node5233 : Graph := .node 5233 row5233 node5231 node5261
set_option maxRecDepth 4096 in
theorem node5233_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5233 :=
  Budget.node_covers _ _ 5233 row5233 node5231 node5261
    node5231_covers node5261_covers (by decide) (by decide) row5233_covers
theorem node5233_eq_data : node5233 = Data.node5233 := by
  have branches : (node5231, node5261) = (Data.node5231, Data.node5261) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5231_eq_data node5261_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5233 row children.1 children.2) row5233_eq_data branches
set_option maxRecDepth 4096 in
theorem row5279_eq_data : row5279 = Data.row5279 := rfl
def node5279 : Graph := .node 5279 row5279 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5279_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5279 :=
  Budget.node_covers _ _ 5279 row5279 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5279_covers
theorem node5279_eq_data : node5279 = Data.node5279 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5279 row children.1 children.2) row5279_eq_data branches
set_option maxRecDepth 4096 in
theorem row5281_eq_data : row5281 = Data.row5281 := rfl
def node5281 : Graph := .node 5281 row5281 node5279 Graph.empty
set_option maxRecDepth 4096 in
theorem node5281_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5281 :=
  Budget.node_covers _ _ 5281 row5281 node5279 Graph.empty
    node5279_covers empty_covers (by decide) (by decide) row5281_covers
theorem node5281_eq_data : node5281 = Data.node5281 := by
  have branches : (node5279, Graph.empty) = (Data.node5279, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5279_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5281 row children.1 children.2) row5281_eq_data branches
set_option maxRecDepth 4096 in
theorem row5303_eq_data : row5303 = Data.row5303 := rfl
def node5303 : Graph := .node 5303 row5303 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5303_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5303 :=
  Budget.node_covers _ _ 5303 row5303 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5303_covers
theorem node5303_eq_data : node5303 = Data.node5303 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5303 row children.1 children.2) row5303_eq_data branches
set_option maxRecDepth 4096 in
theorem row5309_eq_data : row5309 = Data.row5309 := rfl
def node5309 : Graph := .node 5309 row5309 node5303 Graph.empty
set_option maxRecDepth 4096 in
theorem node5309_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5309 :=
  Budget.node_covers _ _ 5309 row5309 node5303 Graph.empty
    node5303_covers empty_covers (by decide) (by decide) row5309_covers
theorem node5309_eq_data : node5309 = Data.node5309 := by
  have branches : (node5303, Graph.empty) = (Data.node5303, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5303_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5309 row children.1 children.2) row5309_eq_data branches
set_option maxRecDepth 4096 in
theorem row5297_eq_data : row5297 = Data.row5297 := rfl
def node5297 : Graph := .node 5297 row5297 node5281 node5309
set_option maxRecDepth 4096 in
theorem node5297_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5297 :=
  Budget.node_covers _ _ 5297 row5297 node5281 node5309
    node5281_covers node5309_covers (by decide) (by decide) row5297_covers
theorem node5297_eq_data : node5297 = Data.node5297 := by
  have branches : (node5281, node5309) = (Data.node5281, Data.node5309) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5281_eq_data node5309_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5297 row children.1 children.2) row5297_eq_data branches
set_option maxRecDepth 4096 in
theorem row5273_eq_data : row5273 = Data.row5273 := rfl
def node5273 : Graph := .node 5273 row5273 node5233 node5297
set_option maxRecDepth 4096 in
theorem node5273_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5273 :=
  Budget.node_covers _ _ 5273 row5273 node5233 node5297
    node5233_covers node5297_covers (by decide) (by decide) row5273_covers
theorem node5273_eq_data : node5273 = Data.node5273 := by
  have branches : (node5233, node5297) = (Data.node5233, Data.node5297) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5233_eq_data node5297_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5273 row children.1 children.2) row5273_eq_data branches
set_option maxRecDepth 4096 in
theorem row5209_eq_data : row5209 = Data.row5209 := rfl
def node5209 : Graph := .node 5209 row5209 node5153 node5273
set_option maxRecDepth 4096 in
theorem node5209_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5209 :=
  Budget.node_covers _ _ 5209 row5209 node5153 node5273
    node5153_covers node5273_covers (by decide) (by decide) row5209_covers
theorem node5209_eq_data : node5209 = Data.node5209 := by
  have branches : (node5153, node5273) = (Data.node5153, Data.node5273) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5153_eq_data node5273_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5209 row children.1 children.2) row5209_eq_data branches
set_option maxRecDepth 4096 in
theorem row5333_eq_data : row5333 = Data.row5333 := rfl
def node5333 : Graph := .node 5333 row5333 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5333_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5333 :=
  Budget.node_covers _ _ 5333 row5333 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5333_covers
theorem node5333_eq_data : node5333 = Data.node5333 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5333 row children.1 children.2) row5333_eq_data branches
set_option maxRecDepth 4096 in
theorem row5347_eq_data : row5347 = Data.row5347 := rfl
def node5347 : Graph := .node 5347 row5347 node5333 Graph.empty
set_option maxRecDepth 4096 in
theorem node5347_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5347 :=
  Budget.node_covers _ _ 5347 row5347 node5333 Graph.empty
    node5333_covers empty_covers (by decide) (by decide) row5347_covers
theorem node5347_eq_data : node5347 = Data.node5347 := by
  have branches : (node5333, Graph.empty) = (Data.node5333, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5333_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5347 row children.1 children.2) row5347_eq_data branches
set_option maxRecDepth 4096 in
theorem row5381_eq_data : row5381 = Data.row5381 := rfl
def node5381 : Graph := .node 5381 row5381 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5381_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5381 :=
  Budget.node_covers _ _ 5381 row5381 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5381_covers
theorem node5381_eq_data : node5381 = Data.node5381 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5381 row children.1 children.2) row5381_eq_data branches
set_option maxRecDepth 4096 in
theorem row5387_eq_data : row5387 = Data.row5387 := rfl
def node5387 : Graph := .node 5387 row5387 node5381 Graph.empty
set_option maxRecDepth 4096 in
theorem node5387_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5387 :=
  Budget.node_covers _ _ 5387 row5387 node5381 Graph.empty
    node5381_covers empty_covers (by decide) (by decide) row5387_covers
theorem node5387_eq_data : node5387 = Data.node5387 := by
  have branches : (node5381, Graph.empty) = (Data.node5381, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5381_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5387 row children.1 children.2) row5387_eq_data branches
set_option maxRecDepth 4096 in
theorem row5351_eq_data : row5351 = Data.row5351 := rfl
def node5351 : Graph := .node 5351 row5351 node5347 node5387
set_option maxRecDepth 4096 in
theorem node5351_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5351 :=
  Budget.node_covers _ _ 5351 row5351 node5347 node5387
    node5347_covers node5387_covers (by decide) (by decide) row5351_covers
theorem node5351_eq_data : node5351 = Data.node5351 := by
  have branches : (node5347, node5387) = (Data.node5347, Data.node5387) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5347_eq_data node5387_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5351 row children.1 children.2) row5351_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
