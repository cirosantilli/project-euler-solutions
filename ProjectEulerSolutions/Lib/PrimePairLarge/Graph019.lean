import ProjectEulerSolutions.Lib.PrimePairLarge.Graph018
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row9929_eq_data : row9929 = Data.row9929 := rfl
def node9929 : Graph := .node 9929 row9929 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9929_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9929 :=
  Budget.node_covers _ _ 9929 row9929 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9929_covers
theorem node9929_eq_data : node9929 = Data.node9929 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9929 row children.1 children.2) row9929_eq_data branches
set_option maxRecDepth 4096 in
theorem row9931_eq_data : row9931 = Data.row9931 := rfl
def node9931 : Graph := .node 9931 row9931 node9929 Graph.empty
set_option maxRecDepth 4096 in
theorem node9931_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9931 :=
  Budget.node_covers _ _ 9931 row9931 node9929 Graph.empty
    node9929_covers empty_covers (by decide) (by decide) row9931_covers
theorem node9931_eq_data : node9931 = Data.node9931 := by
  have branches : (node9929, Graph.empty) = (Data.node9929, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9929_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9931 row children.1 children.2) row9931_eq_data branches
set_option maxRecDepth 4096 in
theorem row9923_eq_data : row9923 = Data.row9923 := rfl
def node9923 : Graph := .node 9923 row9923 node9907 node9931
set_option maxRecDepth 4096 in
theorem node9923_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9923 :=
  Budget.node_covers _ _ 9923 row9923 node9907 node9931
    node9907_covers node9931_covers (by decide) (by decide) row9923_covers
theorem node9923_eq_data : node9923 = Data.node9923 := by
  have branches : (node9907, node9931) = (Data.node9907, Data.node9931) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9907_eq_data node9931_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9923 row children.1 children.2) row9923_eq_data branches
set_option maxRecDepth 4096 in
theorem row9887_eq_data : row9887 = Data.row9887 := rfl
def node9887 : Graph := .node 9887 row9887 node9859 node9923
set_option maxRecDepth 4096 in
theorem node9887_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9887 :=
  Budget.node_covers _ _ 9887 row9887 node9859 node9923
    node9859_covers node9923_covers (by decide) (by decide) row9887_covers
theorem node9887_eq_data : node9887 = Data.node9887 := by
  have branches : (node9859, node9923) = (Data.node9859, Data.node9923) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9859_eq_data node9923_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9887 row children.1 children.2) row9887_eq_data branches
set_option maxRecDepth 4096 in
theorem row9949_eq_data : row9949 = Data.row9949 := rfl
def node9949 : Graph := .node 9949 row9949 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9949_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9949 :=
  Budget.node_covers _ _ 9949 row9949 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9949_covers
theorem node9949_eq_data : node9949 = Data.node9949 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9949 row children.1 children.2) row9949_eq_data branches
set_option maxRecDepth 4096 in
theorem row9967_eq_data : row9967 = Data.row9967 := rfl
def node9967 : Graph := .node 9967 row9967 node9949 Graph.empty
set_option maxRecDepth 4096 in
theorem node9967_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9967 :=
  Budget.node_covers _ _ 9967 row9967 node9949 Graph.empty
    node9949_covers empty_covers (by decide) (by decide) row9967_covers
theorem node9967_eq_data : node9967 = Data.node9967 := by
  have branches : (node9949, Graph.empty) = (Data.node9949, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9949_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9967 row children.1 children.2) row9967_eq_data branches
set_option maxRecDepth 4096 in
theorem row10007_eq_data : row10007 = Data.row10007 := rfl
def node10007 : Graph := .node 10007 row10007 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10007_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10007 :=
  Budget.node_covers _ _ 10007 row10007 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10007_covers
theorem node10007_eq_data : node10007 = Data.node10007 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10007 row children.1 children.2) row10007_eq_data branches
set_option maxRecDepth 4096 in
theorem row10009_eq_data : row10009 = Data.row10009 := rfl
def node10009 : Graph := .node 10009 row10009 node10007 Graph.empty
set_option maxRecDepth 4096 in
theorem node10009_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10009 :=
  Budget.node_covers _ _ 10009 row10009 node10007 Graph.empty
    node10007_covers empty_covers (by decide) (by decide) row10009_covers
theorem node10009_eq_data : node10009 = Data.node10009 := by
  have branches : (node10007, Graph.empty) = (Data.node10007, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10007_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10009 row children.1 children.2) row10009_eq_data branches
set_option maxRecDepth 4096 in
theorem row9973_eq_data : row9973 = Data.row9973 := rfl
def node9973 : Graph := .node 9973 row9973 node9967 node10009
set_option maxRecDepth 4096 in
theorem node9973_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9973 :=
  Budget.node_covers _ _ 9973 row9973 node9967 node10009
    node9967_covers node10009_covers (by decide) (by decide) row9973_covers
theorem node9973_eq_data : node9973 = Data.node9973 := by
  have branches : (node9967, node10009) = (Data.node9967, Data.node10009) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9967_eq_data node10009_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9973 row children.1 children.2) row9973_eq_data branches
set_option maxRecDepth 4096 in
theorem row10039_eq_data : row10039 = Data.row10039 := rfl
def node10039 : Graph := .node 10039 row10039 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10039_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10039 :=
  Budget.node_covers _ _ 10039 row10039 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10039_covers
theorem node10039_eq_data : node10039 = Data.node10039 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10039 row children.1 children.2) row10039_eq_data branches
set_option maxRecDepth 4096 in
theorem row10061_eq_data : row10061 = Data.row10061 := rfl
def node10061 : Graph := .node 10061 row10061 node10039 Graph.empty
set_option maxRecDepth 4096 in
theorem node10061_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10061 :=
  Budget.node_covers _ _ 10061 row10061 node10039 Graph.empty
    node10039_covers empty_covers (by decide) (by decide) row10061_covers
theorem node10061_eq_data : node10061 = Data.node10061 := by
  have branches : (node10039, Graph.empty) = (Data.node10039, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10039_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10061 row children.1 children.2) row10061_eq_data branches
set_option maxRecDepth 4096 in
theorem row10069_eq_data : row10069 = Data.row10069 := rfl
def node10069 : Graph := .node 10069 row10069 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10069_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10069 :=
  Budget.node_covers _ _ 10069 row10069 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10069_covers
theorem node10069_eq_data : node10069 = Data.node10069 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10069 row children.1 children.2) row10069_eq_data branches
set_option maxRecDepth 4096 in
theorem row10079_eq_data : row10079 = Data.row10079 := rfl
def node10079 : Graph := .node 10079 row10079 node10069 Graph.empty
set_option maxRecDepth 4096 in
theorem node10079_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10079 :=
  Budget.node_covers _ _ 10079 row10079 node10069 Graph.empty
    node10069_covers empty_covers (by decide) (by decide) row10079_covers
theorem node10079_eq_data : node10079 = Data.node10079 := by
  have branches : (node10069, Graph.empty) = (Data.node10069, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10069_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10079 row children.1 children.2) row10079_eq_data branches
set_option maxRecDepth 4096 in
theorem row10067_eq_data : row10067 = Data.row10067 := rfl
def node10067 : Graph := .node 10067 row10067 node10061 node10079
set_option maxRecDepth 4096 in
theorem node10067_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10067 :=
  Budget.node_covers _ _ 10067 row10067 node10061 node10079
    node10061_covers node10079_covers (by decide) (by decide) row10067_covers
theorem node10067_eq_data : node10067 = Data.node10067 := by
  have branches : (node10061, node10079) = (Data.node10061, Data.node10079) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10061_eq_data node10079_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10067 row children.1 children.2) row10067_eq_data branches
set_option maxRecDepth 4096 in
theorem row10037_eq_data : row10037 = Data.row10037 := rfl
def node10037 : Graph := .node 10037 row10037 node9973 node10067
set_option maxRecDepth 4096 in
theorem node10037_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10037 :=
  Budget.node_covers _ _ 10037 row10037 node9973 node10067
    node9973_covers node10067_covers (by decide) (by decide) row10037_covers
theorem node10037_eq_data : node10037 = Data.node10037 := by
  have branches : (node9973, node10067) = (Data.node9973, Data.node10067) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9973_eq_data node10067_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10037 row children.1 children.2) row10037_eq_data branches
set_option maxRecDepth 4096 in
theorem row9941_eq_data : row9941 = Data.row9941 := rfl
def node9941 : Graph := .node 9941 row9941 node9887 node10037
set_option maxRecDepth 4096 in
theorem node9941_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9941 :=
  Budget.node_covers _ _ 9941 row9941 node9887 node10037
    node9887_covers node10037_covers (by decide) (by decide) row9941_covers
theorem node9941_eq_data : node9941 = Data.node9941 := by
  have branches : (node9887, node10037) = (Data.node9887, Data.node10037) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9887_eq_data node10037_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9941 row children.1 children.2) row9941_eq_data branches
set_option maxRecDepth 4096 in
theorem row10093_eq_data : row10093 = Data.row10093 := rfl
def node10093 : Graph := .node 10093 row10093 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10093_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10093 :=
  Budget.node_covers _ _ 10093 row10093 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10093_covers
theorem node10093_eq_data : node10093 = Data.node10093 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10093 row children.1 children.2) row10093_eq_data branches
set_option maxRecDepth 4096 in
theorem row10099_eq_data : row10099 = Data.row10099 := rfl
def node10099 : Graph := .node 10099 row10099 node10093 Graph.empty
set_option maxRecDepth 4096 in
theorem node10099_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10099 :=
  Budget.node_covers _ _ 10099 row10099 node10093 Graph.empty
    node10093_covers empty_covers (by decide) (by decide) row10099_covers
theorem node10099_eq_data : node10099 = Data.node10099 := by
  have branches : (node10093, Graph.empty) = (Data.node10093, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10093_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10099 row children.1 children.2) row10099_eq_data branches
set_option maxRecDepth 4096 in
theorem row10111_eq_data : row10111 = Data.row10111 := rfl
def node10111 : Graph := .node 10111 row10111 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10111_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10111 :=
  Budget.node_covers _ _ 10111 row10111 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10111_covers
theorem node10111_eq_data : node10111 = Data.node10111 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10111 row children.1 children.2) row10111_eq_data branches
set_option maxRecDepth 4096 in
theorem row10133_eq_data : row10133 = Data.row10133 := rfl
def node10133 : Graph := .node 10133 row10133 node10111 Graph.empty
set_option maxRecDepth 4096 in
theorem node10133_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10133 :=
  Budget.node_covers _ _ 10133 row10133 node10111 Graph.empty
    node10111_covers empty_covers (by decide) (by decide) row10133_covers
theorem node10133_eq_data : node10133 = Data.node10133 := by
  have branches : (node10111, Graph.empty) = (Data.node10111, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10111_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10133 row children.1 children.2) row10133_eq_data branches
set_option maxRecDepth 4096 in
theorem row10103_eq_data : row10103 = Data.row10103 := rfl
def node10103 : Graph := .node 10103 row10103 node10099 node10133
set_option maxRecDepth 4096 in
theorem node10103_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10103 :=
  Budget.node_covers _ _ 10103 row10103 node10099 node10133
    node10099_covers node10133_covers (by decide) (by decide) row10103_covers
theorem node10103_eq_data : node10103 = Data.node10103 := by
  have branches : (node10099, node10133) = (Data.node10099, Data.node10133) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10099_eq_data node10133_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10103 row children.1 children.2) row10103_eq_data branches
set_option maxRecDepth 4096 in
theorem row10141_eq_data : row10141 = Data.row10141 := rfl
def node10141 : Graph := .node 10141 row10141 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10141_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10141 :=
  Budget.node_covers _ _ 10141 row10141 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10141_covers
theorem node10141_eq_data : node10141 = Data.node10141 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10141 row children.1 children.2) row10141_eq_data branches
set_option maxRecDepth 4096 in
theorem row10151_eq_data : row10151 = Data.row10151 := rfl
def node10151 : Graph := .node 10151 row10151 node10141 Graph.empty
set_option maxRecDepth 4096 in
theorem node10151_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10151 :=
  Budget.node_covers _ _ 10151 row10151 node10141 Graph.empty
    node10141_covers empty_covers (by decide) (by decide) row10151_covers
theorem node10151_eq_data : node10151 = Data.node10151 := by
  have branches : (node10141, Graph.empty) = (Data.node10141, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10141_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10151 row children.1 children.2) row10151_eq_data branches
set_option maxRecDepth 4096 in
theorem row10163_eq_data : row10163 = Data.row10163 := rfl
def node10163 : Graph := .node 10163 row10163 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10163_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10163 :=
  Budget.node_covers _ _ 10163 row10163 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10163_covers
theorem node10163_eq_data : node10163 = Data.node10163 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10163 row children.1 children.2) row10163_eq_data branches
set_option maxRecDepth 4096 in
theorem row10169_eq_data : row10169 = Data.row10169 := rfl
def node10169 : Graph := .node 10169 row10169 node10163 Graph.empty
set_option maxRecDepth 4096 in
theorem node10169_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10169 :=
  Budget.node_covers _ _ 10169 row10169 node10163 Graph.empty
    node10163_covers empty_covers (by decide) (by decide) row10169_covers
theorem node10169_eq_data : node10169 = Data.node10169 := by
  have branches : (node10163, Graph.empty) = (Data.node10163, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10163_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10169 row children.1 children.2) row10169_eq_data branches
set_option maxRecDepth 4096 in
theorem row10159_eq_data : row10159 = Data.row10159 := rfl
def node10159 : Graph := .node 10159 row10159 node10151 node10169
set_option maxRecDepth 4096 in
theorem node10159_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10159 :=
  Budget.node_covers _ _ 10159 row10159 node10151 node10169
    node10151_covers node10169_covers (by decide) (by decide) row10159_covers
theorem node10159_eq_data : node10159 = Data.node10159 := by
  have branches : (node10151, node10169) = (Data.node10151, Data.node10169) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10151_eq_data node10169_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10159 row children.1 children.2) row10159_eq_data branches
set_option maxRecDepth 4096 in
theorem row10139_eq_data : row10139 = Data.row10139 := rfl
def node10139 : Graph := .node 10139 row10139 node10103 node10159
set_option maxRecDepth 4096 in
theorem node10139_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10139 :=
  Budget.node_covers _ _ 10139 row10139 node10103 node10159
    node10103_covers node10159_covers (by decide) (by decide) row10139_covers
theorem node10139_eq_data : node10139 = Data.node10139 := by
  have branches : (node10103, node10159) = (Data.node10103, Data.node10159) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10103_eq_data node10159_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10139 row children.1 children.2) row10139_eq_data branches
set_option maxRecDepth 4096 in
theorem row10181_eq_data : row10181 = Data.row10181 := rfl
def node10181 : Graph := .node 10181 row10181 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10181_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10181 :=
  Budget.node_covers _ _ 10181 row10181 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10181_covers
theorem node10181_eq_data : node10181 = Data.node10181 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10181 row children.1 children.2) row10181_eq_data branches
set_option maxRecDepth 4096 in
theorem row10193_eq_data : row10193 = Data.row10193 := rfl
def node10193 : Graph := .node 10193 row10193 node10181 Graph.empty
set_option maxRecDepth 4096 in
theorem node10193_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10193 :=
  Budget.node_covers _ _ 10193 row10193 node10181 Graph.empty
    node10181_covers empty_covers (by decide) (by decide) row10193_covers
theorem node10193_eq_data : node10193 = Data.node10193 := by
  have branches : (node10181, Graph.empty) = (Data.node10181, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10181_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10193 row children.1 children.2) row10193_eq_data branches
set_option maxRecDepth 4096 in
theorem row10223_eq_data : row10223 = Data.row10223 := rfl
def node10223 : Graph := .node 10223 row10223 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10223_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10223 :=
  Budget.node_covers _ _ 10223 row10223 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10223_covers
theorem node10223_eq_data : node10223 = Data.node10223 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10223 row children.1 children.2) row10223_eq_data branches
set_option maxRecDepth 4096 in
theorem row10243_eq_data : row10243 = Data.row10243 := rfl
def node10243 : Graph := .node 10243 row10243 node10223 Graph.empty
set_option maxRecDepth 4096 in
theorem node10243_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10243 :=
  Budget.node_covers _ _ 10243 row10243 node10223 Graph.empty
    node10223_covers empty_covers (by decide) (by decide) row10243_covers
theorem node10243_eq_data : node10243 = Data.node10243 := by
  have branches : (node10223, Graph.empty) = (Data.node10223, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10223_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10243 row children.1 children.2) row10243_eq_data branches
set_option maxRecDepth 4096 in
theorem row10211_eq_data : row10211 = Data.row10211 := rfl
def node10211 : Graph := .node 10211 row10211 node10193 node10243
set_option maxRecDepth 4096 in
theorem node10211_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10211 :=
  Budget.node_covers _ _ 10211 row10211 node10193 node10243
    node10193_covers node10243_covers (by decide) (by decide) row10211_covers
theorem node10211_eq_data : node10211 = Data.node10211 := by
  have branches : (node10193, node10243) = (Data.node10193, Data.node10243) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10193_eq_data node10243_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10211 row children.1 children.2) row10211_eq_data branches
set_option maxRecDepth 4096 in
theorem row10253_eq_data : row10253 = Data.row10253 := rfl
def node10253 : Graph := .node 10253 row10253 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10253_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10253 :=
  Budget.node_covers _ _ 10253 row10253 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10253_covers
theorem node10253_eq_data : node10253 = Data.node10253 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10253 row children.1 children.2) row10253_eq_data branches
set_option maxRecDepth 4096 in
theorem row10259_eq_data : row10259 = Data.row10259 := rfl
def node10259 : Graph := .node 10259 row10259 node10253 Graph.empty
set_option maxRecDepth 4096 in
theorem node10259_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10259 :=
  Budget.node_covers _ _ 10259 row10259 node10253 Graph.empty
    node10253_covers empty_covers (by decide) (by decide) row10259_covers
theorem node10259_eq_data : node10259 = Data.node10259 := by
  have branches : (node10253, Graph.empty) = (Data.node10253, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10253_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10259 row children.1 children.2) row10259_eq_data branches
set_option maxRecDepth 4096 in
theorem row10271_eq_data : row10271 = Data.row10271 := rfl
def node10271 : Graph := .node 10271 row10271 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10271_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10271 :=
  Budget.node_covers _ _ 10271 row10271 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10271_covers
theorem node10271_eq_data : node10271 = Data.node10271 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10271 row children.1 children.2) row10271_eq_data branches
set_option maxRecDepth 4096 in
theorem row10273_eq_data : row10273 = Data.row10273 := rfl
def node10273 : Graph := .node 10273 row10273 node10271 Graph.empty
set_option maxRecDepth 4096 in
theorem node10273_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10273 :=
  Budget.node_covers _ _ 10273 row10273 node10271 Graph.empty
    node10271_covers empty_covers (by decide) (by decide) row10273_covers
theorem node10273_eq_data : node10273 = Data.node10273 := by
  have branches : (node10271, Graph.empty) = (Data.node10271, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10271_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10273 row children.1 children.2) row10273_eq_data branches
set_option maxRecDepth 4096 in
theorem row10267_eq_data : row10267 = Data.row10267 := rfl
def node10267 : Graph := .node 10267 row10267 node10259 node10273
set_option maxRecDepth 4096 in
theorem node10267_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10267 :=
  Budget.node_covers _ _ 10267 row10267 node10259 node10273
    node10259_covers node10273_covers (by decide) (by decide) row10267_covers
theorem node10267_eq_data : node10267 = Data.node10267 := by
  have branches : (node10259, node10273) = (Data.node10259, Data.node10273) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10259_eq_data node10273_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10267 row children.1 children.2) row10267_eq_data branches
set_option maxRecDepth 4096 in
theorem row10247_eq_data : row10247 = Data.row10247 := rfl
def node10247 : Graph := .node 10247 row10247 node10211 node10267
set_option maxRecDepth 4096 in
theorem node10247_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10247 :=
  Budget.node_covers _ _ 10247 row10247 node10211 node10267
    node10211_covers node10267_covers (by decide) (by decide) row10247_covers
theorem node10247_eq_data : node10247 = Data.node10247 := by
  have branches : (node10211, node10267) = (Data.node10211, Data.node10267) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10211_eq_data node10267_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10247 row children.1 children.2) row10247_eq_data branches
set_option maxRecDepth 4096 in
theorem row10177_eq_data : row10177 = Data.row10177 := rfl
def node10177 : Graph := .node 10177 row10177 node10139 node10247
set_option maxRecDepth 4096 in
theorem node10177_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10177 :=
  Budget.node_covers _ _ 10177 row10177 node10139 node10247
    node10139_covers node10247_covers (by decide) (by decide) row10177_covers
theorem node10177_eq_data : node10177 = Data.node10177 := by
  have branches : (node10139, node10247) = (Data.node10139, Data.node10247) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10139_eq_data node10247_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10177 row children.1 children.2) row10177_eq_data branches
set_option maxRecDepth 4096 in
theorem row10091_eq_data : row10091 = Data.row10091 := rfl
def node10091 : Graph := .node 10091 row10091 node9941 node10177
set_option maxRecDepth 4096 in
theorem node10091_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10091 :=
  Budget.node_covers _ _ 10091 row10091 node9941 node10177
    node9941_covers node10177_covers (by decide) (by decide) row10091_covers
theorem node10091_eq_data : node10091 = Data.node10091 := by
  have branches : (node9941, node10177) = (Data.node9941, Data.node10177) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9941_eq_data node10177_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10091 row children.1 children.2) row10091_eq_data branches
set_option maxRecDepth 4096 in
theorem row9839_eq_data : row9839 = Data.row9839 := rfl
def node9839 : Graph := .node 9839 row9839 node9631 node10091
set_option maxRecDepth 4096 in
theorem node9839_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9839 :=
  Budget.node_covers _ _ 9839 row9839 node9631 node10091
    node9631_covers node10091_covers (by decide) (by decide) row9839_covers
theorem node9839_eq_data : node9839 = Data.node9839 := by
  have branches : (node9631, node10091) = (Data.node9631, Data.node10091) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9631_eq_data node10091_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9839 row children.1 children.2) row9839_eq_data branches
set_option maxRecDepth 4096 in
theorem row10301_eq_data : row10301 = Data.row10301 := rfl
def node10301 : Graph := .node 10301 row10301 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10301_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10301 :=
  Budget.node_covers _ _ 10301 row10301 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10301_covers
theorem node10301_eq_data : node10301 = Data.node10301 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10301 row children.1 children.2) row10301_eq_data branches
set_option maxRecDepth 4096 in
theorem row10313_eq_data : row10313 = Data.row10313 := rfl
def node10313 : Graph := .node 10313 row10313 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10313_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10313 :=
  Budget.node_covers _ _ 10313 row10313 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10313_covers
theorem node10313_eq_data : node10313 = Data.node10313 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10313 row children.1 children.2) row10313_eq_data branches
set_option maxRecDepth 4096 in
theorem row10303_eq_data : row10303 = Data.row10303 := rfl
def node10303 : Graph := .node 10303 row10303 node10301 node10313
set_option maxRecDepth 4096 in
theorem node10303_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10303 :=
  Budget.node_covers _ _ 10303 row10303 node10301 node10313
    node10301_covers node10313_covers (by decide) (by decide) row10303_covers
theorem node10303_eq_data : node10303 = Data.node10303 := by
  have branches : (node10301, node10313) = (Data.node10301, Data.node10313) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10301_eq_data node10313_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10303 row children.1 children.2) row10303_eq_data branches
set_option maxRecDepth 4096 in
theorem row10331_eq_data : row10331 = Data.row10331 := rfl
def node10331 : Graph := .node 10331 row10331 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10331_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10331 :=
  Budget.node_covers _ _ 10331 row10331 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10331_covers
theorem node10331_eq_data : node10331 = Data.node10331 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10331 row children.1 children.2) row10331_eq_data branches
set_option maxRecDepth 4096 in
theorem row10333_eq_data : row10333 = Data.row10333 := rfl
def node10333 : Graph := .node 10333 row10333 node10331 Graph.empty
set_option maxRecDepth 4096 in
theorem node10333_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10333 :=
  Budget.node_covers _ _ 10333 row10333 node10331 Graph.empty
    node10331_covers empty_covers (by decide) (by decide) row10333_covers
theorem node10333_eq_data : node10333 = Data.node10333 := by
  have branches : (node10331, Graph.empty) = (Data.node10331, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10331_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10333 row children.1 children.2) row10333_eq_data branches
set_option maxRecDepth 4096 in
theorem row10321_eq_data : row10321 = Data.row10321 := rfl
def node10321 : Graph := .node 10321 row10321 node10303 node10333
set_option maxRecDepth 4096 in
theorem node10321_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10321 :=
  Budget.node_covers _ _ 10321 row10321 node10303 node10333
    node10303_covers node10333_covers (by decide) (by decide) row10321_covers
theorem node10321_eq_data : node10321 = Data.node10321 := by
  have branches : (node10303, node10333) = (Data.node10303, Data.node10333) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10303_eq_data node10333_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10321 row children.1 children.2) row10321_eq_data branches
set_option maxRecDepth 4096 in
theorem row10343_eq_data : row10343 = Data.row10343 := rfl
def node10343 : Graph := .node 10343 row10343 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10343_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10343 :=
  Budget.node_covers _ _ 10343 row10343 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10343_covers
theorem node10343_eq_data : node10343 = Data.node10343 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10343 row children.1 children.2) row10343_eq_data branches
set_option maxRecDepth 4096 in
theorem row10357_eq_data : row10357 = Data.row10357 := rfl
def node10357 : Graph := .node 10357 row10357 node10343 Graph.empty
set_option maxRecDepth 4096 in
theorem node10357_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10357 :=
  Budget.node_covers _ _ 10357 row10357 node10343 Graph.empty
    node10343_covers empty_covers (by decide) (by decide) row10357_covers
theorem node10357_eq_data : node10357 = Data.node10357 := by
  have branches : (node10343, Graph.empty) = (Data.node10343, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10343_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10357 row children.1 children.2) row10357_eq_data branches
set_option maxRecDepth 4096 in
theorem row10391_eq_data : row10391 = Data.row10391 := rfl
def node10391 : Graph := .node 10391 row10391 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10391_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10391 :=
  Budget.node_covers _ _ 10391 row10391 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10391_covers
theorem node10391_eq_data : node10391 = Data.node10391 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10391 row children.1 children.2) row10391_eq_data branches
set_option maxRecDepth 4096 in
theorem row10399_eq_data : row10399 = Data.row10399 := rfl
def node10399 : Graph := .node 10399 row10399 node10391 Graph.empty
set_option maxRecDepth 4096 in
theorem node10399_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10399 :=
  Budget.node_covers _ _ 10399 row10399 node10391 Graph.empty
    node10391_covers empty_covers (by decide) (by decide) row10399_covers
theorem node10399_eq_data : node10399 = Data.node10399 := by
  have branches : (node10391, Graph.empty) = (Data.node10391, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10391_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10399 row children.1 children.2) row10399_eq_data branches
set_option maxRecDepth 4096 in
theorem row10369_eq_data : row10369 = Data.row10369 := rfl
def node10369 : Graph := .node 10369 row10369 node10357 node10399
set_option maxRecDepth 4096 in
theorem node10369_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10369 :=
  Budget.node_covers _ _ 10369 row10369 node10357 node10399
    node10357_covers node10399_covers (by decide) (by decide) row10369_covers
theorem node10369_eq_data : node10369 = Data.node10369 := by
  have branches : (node10357, node10399) = (Data.node10357, Data.node10399) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10357_eq_data node10399_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10369 row children.1 children.2) row10369_eq_data branches
set_option maxRecDepth 4096 in
theorem row10337_eq_data : row10337 = Data.row10337 := rfl
def node10337 : Graph := .node 10337 row10337 node10321 node10369
set_option maxRecDepth 4096 in
theorem node10337_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10337 :=
  Budget.node_covers _ _ 10337 row10337 node10321 node10369
    node10321_covers node10369_covers (by decide) (by decide) row10337_covers
theorem node10337_eq_data : node10337 = Data.node10337 := by
  have branches : (node10321, node10369) = (Data.node10321, Data.node10369) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10321_eq_data node10369_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10337 row children.1 children.2) row10337_eq_data branches
set_option maxRecDepth 4096 in
theorem row10429_eq_data : row10429 = Data.row10429 := rfl
def node10429 : Graph := .node 10429 row10429 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10429_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10429 :=
  Budget.node_covers _ _ 10429 row10429 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10429_covers
theorem node10429_eq_data : node10429 = Data.node10429 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10429 row children.1 children.2) row10429_eq_data branches
set_option maxRecDepth 4096 in
theorem row10433_eq_data : row10433 = Data.row10433 := rfl
def node10433 : Graph := .node 10433 row10433 node10429 Graph.empty
set_option maxRecDepth 4096 in
theorem node10433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10433 :=
  Budget.node_covers _ _ 10433 row10433 node10429 Graph.empty
    node10429_covers empty_covers (by decide) (by decide) row10433_covers
theorem node10433_eq_data : node10433 = Data.node10433 := by
  have branches : (node10429, Graph.empty) = (Data.node10429, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10429_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10433 row children.1 children.2) row10433_eq_data branches
set_option maxRecDepth 4096 in
theorem row10457_eq_data : row10457 = Data.row10457 := rfl
def node10457 : Graph := .node 10457 row10457 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10457_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10457 :=
  Budget.node_covers _ _ 10457 row10457 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10457_covers
theorem node10457_eq_data : node10457 = Data.node10457 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10457 row children.1 children.2) row10457_eq_data branches
set_option maxRecDepth 4096 in
theorem row10459_eq_data : row10459 = Data.row10459 := rfl
def node10459 : Graph := .node 10459 row10459 node10457 Graph.empty
set_option maxRecDepth 4096 in
theorem node10459_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10459 :=
  Budget.node_covers _ _ 10459 row10459 node10457 Graph.empty
    node10457_covers empty_covers (by decide) (by decide) row10459_covers
theorem node10459_eq_data : node10459 = Data.node10459 := by
  have branches : (node10457, Graph.empty) = (Data.node10457, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10457_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10459 row children.1 children.2) row10459_eq_data branches
set_option maxRecDepth 4096 in
theorem row10453_eq_data : row10453 = Data.row10453 := rfl
def node10453 : Graph := .node 10453 row10453 node10433 node10459
set_option maxRecDepth 4096 in
theorem node10453_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10453 :=
  Budget.node_covers _ _ 10453 row10453 node10433 node10459
    node10433_covers node10459_covers (by decide) (by decide) row10453_covers
theorem node10453_eq_data : node10453 = Data.node10453 := by
  have branches : (node10433, node10459) = (Data.node10433, Data.node10459) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10433_eq_data node10459_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10453 row children.1 children.2) row10453_eq_data branches
set_option maxRecDepth 4096 in
theorem row10477_eq_data : row10477 = Data.row10477 := rfl
def node10477 : Graph := .node 10477 row10477 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10477_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10477 :=
  Budget.node_covers _ _ 10477 row10477 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10477_covers
theorem node10477_eq_data : node10477 = Data.node10477 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10477 row children.1 children.2) row10477_eq_data branches
set_option maxRecDepth 4096 in
theorem row10487_eq_data : row10487 = Data.row10487 := rfl
def node10487 : Graph := .node 10487 row10487 node10477 Graph.empty
set_option maxRecDepth 4096 in
theorem node10487_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10487 :=
  Budget.node_covers _ _ 10487 row10487 node10477 Graph.empty
    node10477_covers empty_covers (by decide) (by decide) row10487_covers
theorem node10487_eq_data : node10487 = Data.node10487 := by
  have branches : (node10477, Graph.empty) = (Data.node10477, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10477_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10487 row children.1 children.2) row10487_eq_data branches
set_option maxRecDepth 4096 in
theorem row10501_eq_data : row10501 = Data.row10501 := rfl
def node10501 : Graph := .node 10501 row10501 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10501_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10501 :=
  Budget.node_covers _ _ 10501 row10501 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10501_covers
theorem node10501_eq_data : node10501 = Data.node10501 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10501 row children.1 children.2) row10501_eq_data branches
set_option maxRecDepth 4096 in
theorem row10513_eq_data : row10513 = Data.row10513 := rfl
def node10513 : Graph := .node 10513 row10513 node10501 Graph.empty
set_option maxRecDepth 4096 in
theorem node10513_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10513 :=
  Budget.node_covers _ _ 10513 row10513 node10501 Graph.empty
    node10501_covers empty_covers (by decide) (by decide) row10513_covers
theorem node10513_eq_data : node10513 = Data.node10513 := by
  have branches : (node10501, Graph.empty) = (Data.node10501, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10501_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10513 row children.1 children.2) row10513_eq_data branches
set_option maxRecDepth 4096 in
theorem row10499_eq_data : row10499 = Data.row10499 := rfl
def node10499 : Graph := .node 10499 row10499 node10487 node10513
set_option maxRecDepth 4096 in
theorem node10499_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10499 :=
  Budget.node_covers _ _ 10499 row10499 node10487 node10513
    node10487_covers node10513_covers (by decide) (by decide) row10499_covers
theorem node10499_eq_data : node10499 = Data.node10499 := by
  have branches : (node10487, node10513) = (Data.node10487, Data.node10513) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10487_eq_data node10513_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10499 row children.1 children.2) row10499_eq_data branches
set_option maxRecDepth 4096 in
theorem row10463_eq_data : row10463 = Data.row10463 := rfl
def node10463 : Graph := .node 10463 row10463 node10453 node10499
set_option maxRecDepth 4096 in
theorem node10463_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10463 :=
  Budget.node_covers _ _ 10463 row10463 node10453 node10499
    node10453_covers node10499_covers (by decide) (by decide) row10463_covers
theorem node10463_eq_data : node10463 = Data.node10463 := by
  have branches : (node10453, node10499) = (Data.node10453, Data.node10499) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10453_eq_data node10499_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10463 row children.1 children.2) row10463_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
