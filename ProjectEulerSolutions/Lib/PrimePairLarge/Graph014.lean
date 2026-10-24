import ProjectEulerSolutions.Lib.PrimePairLarge.Graph013
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row7019_eq_data : row7019 = Data.row7019 := rfl
def node7019 : Graph := .node 7019 row7019 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7019_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7019 :=
  Budget.node_covers _ _ 7019 row7019 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7019_covers
theorem node7019_eq_data : node7019 = Data.node7019 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7019 row children.1 children.2) row7019_eq_data branches
set_option maxRecDepth 4096 in
theorem row7027_eq_data : row7027 = Data.row7027 := rfl
def node7027 : Graph := .node 7027 row7027 node7019 Graph.empty
set_option maxRecDepth 4096 in
theorem node7027_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7027 :=
  Budget.node_covers _ _ 7027 row7027 node7019 Graph.empty
    node7019_covers empty_covers (by decide) (by decide) row7027_covers
theorem node7027_eq_data : node7027 = Data.node7027 := by
  have branches : (node7019, Graph.empty) = (Data.node7019, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7019_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7027 row children.1 children.2) row7027_eq_data branches
set_option maxRecDepth 4096 in
theorem row7013_eq_data : row7013 = Data.row7013 := rfl
def node7013 : Graph := .node 7013 row7013 node7001 node7027
set_option maxRecDepth 4096 in
theorem node7013_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7013 :=
  Budget.node_covers _ _ 7013 row7013 node7001 node7027
    node7001_covers node7027_covers (by decide) (by decide) row7013_covers
theorem node7013_eq_data : node7013 = Data.node7013 := by
  have branches : (node7001, node7027) = (Data.node7001, Data.node7027) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7001_eq_data node7027_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7013 row children.1 children.2) row7013_eq_data branches
set_option maxRecDepth 4096 in
theorem row7043_eq_data : row7043 = Data.row7043 := rfl
def node7043 : Graph := .node 7043 row7043 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7043_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7043 :=
  Budget.node_covers _ _ 7043 row7043 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7043_covers
theorem node7043_eq_data : node7043 = Data.node7043 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7043 row children.1 children.2) row7043_eq_data branches
set_option maxRecDepth 4096 in
theorem row7057_eq_data : row7057 = Data.row7057 := rfl
def node7057 : Graph := .node 7057 row7057 node7043 Graph.empty
set_option maxRecDepth 4096 in
theorem node7057_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7057 :=
  Budget.node_covers _ _ 7057 row7057 node7043 Graph.empty
    node7043_covers empty_covers (by decide) (by decide) row7057_covers
theorem node7057_eq_data : node7057 = Data.node7057 := by
  have branches : (node7043, Graph.empty) = (Data.node7043, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7043_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7057 row children.1 children.2) row7057_eq_data branches
set_option maxRecDepth 4096 in
theorem row7079_eq_data : row7079 = Data.row7079 := rfl
def node7079 : Graph := .node 7079 row7079 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7079_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7079 :=
  Budget.node_covers _ _ 7079 row7079 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7079_covers
theorem node7079_eq_data : node7079 = Data.node7079 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7079 row children.1 children.2) row7079_eq_data branches
set_option maxRecDepth 4096 in
theorem row7103_eq_data : row7103 = Data.row7103 := rfl
def node7103 : Graph := .node 7103 row7103 node7079 Graph.empty
set_option maxRecDepth 4096 in
theorem node7103_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7103 :=
  Budget.node_covers _ _ 7103 row7103 node7079 Graph.empty
    node7079_covers empty_covers (by decide) (by decide) row7103_covers
theorem node7103_eq_data : node7103 = Data.node7103 := by
  have branches : (node7079, Graph.empty) = (Data.node7079, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7079_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7103 row children.1 children.2) row7103_eq_data branches
set_option maxRecDepth 4096 in
theorem row7069_eq_data : row7069 = Data.row7069 := rfl
def node7069 : Graph := .node 7069 row7069 node7057 node7103
set_option maxRecDepth 4096 in
theorem node7069_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7069 :=
  Budget.node_covers _ _ 7069 row7069 node7057 node7103
    node7057_covers node7103_covers (by decide) (by decide) row7069_covers
theorem node7069_eq_data : node7069 = Data.node7069 := by
  have branches : (node7057, node7103) = (Data.node7057, Data.node7103) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7057_eq_data node7103_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7069 row children.1 children.2) row7069_eq_data branches
set_option maxRecDepth 4096 in
theorem row7039_eq_data : row7039 = Data.row7039 := rfl
def node7039 : Graph := .node 7039 row7039 node7013 node7069
set_option maxRecDepth 4096 in
theorem node7039_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7039 :=
  Budget.node_covers _ _ 7039 row7039 node7013 node7069
    node7013_covers node7069_covers (by decide) (by decide) row7039_covers
theorem node7039_eq_data : node7039 = Data.node7039 := by
  have branches : (node7013, node7069) = (Data.node7013, Data.node7069) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7013_eq_data node7069_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7039 row children.1 children.2) row7039_eq_data branches
set_option maxRecDepth 4096 in
theorem row7121_eq_data : row7121 = Data.row7121 := rfl
def node7121 : Graph := .node 7121 row7121 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7121_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7121 :=
  Budget.node_covers _ _ 7121 row7121 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7121_covers
theorem node7121_eq_data : node7121 = Data.node7121 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7121 row children.1 children.2) row7121_eq_data branches
set_option maxRecDepth 4096 in
theorem row7127_eq_data : row7127 = Data.row7127 := rfl
def node7127 : Graph := .node 7127 row7127 node7121 Graph.empty
set_option maxRecDepth 4096 in
theorem node7127_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7127 :=
  Budget.node_covers _ _ 7127 row7127 node7121 Graph.empty
    node7121_covers empty_covers (by decide) (by decide) row7127_covers
theorem node7127_eq_data : node7127 = Data.node7127 := by
  have branches : (node7121, Graph.empty) = (Data.node7121, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7121_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7127 row children.1 children.2) row7127_eq_data branches
set_option maxRecDepth 4096 in
theorem row7151_eq_data : row7151 = Data.row7151 := rfl
def node7151 : Graph := .node 7151 row7151 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7151_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7151 :=
  Budget.node_covers _ _ 7151 row7151 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7151_covers
theorem node7151_eq_data : node7151 = Data.node7151 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7151 row children.1 children.2) row7151_eq_data branches
set_option maxRecDepth 4096 in
theorem row7159_eq_data : row7159 = Data.row7159 := rfl
def node7159 : Graph := .node 7159 row7159 node7151 Graph.empty
set_option maxRecDepth 4096 in
theorem node7159_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7159 :=
  Budget.node_covers _ _ 7159 row7159 node7151 Graph.empty
    node7151_covers empty_covers (by decide) (by decide) row7159_covers
theorem node7159_eq_data : node7159 = Data.node7159 := by
  have branches : (node7151, Graph.empty) = (Data.node7151, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7151_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7159 row children.1 children.2) row7159_eq_data branches
set_option maxRecDepth 4096 in
theorem row7129_eq_data : row7129 = Data.row7129 := rfl
def node7129 : Graph := .node 7129 row7129 node7127 node7159
set_option maxRecDepth 4096 in
theorem node7129_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7129 :=
  Budget.node_covers _ _ 7129 row7129 node7127 node7159
    node7127_covers node7159_covers (by decide) (by decide) row7129_covers
theorem node7129_eq_data : node7129 = Data.node7129 := by
  have branches : (node7127, node7159) = (Data.node7127, Data.node7159) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7127_eq_data node7159_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7129 row children.1 children.2) row7129_eq_data branches
set_option maxRecDepth 4096 in
theorem row7187_eq_data : row7187 = Data.row7187 := rfl
def node7187 : Graph := .node 7187 row7187 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7187_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7187 :=
  Budget.node_covers _ _ 7187 row7187 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7187_covers
theorem node7187_eq_data : node7187 = Data.node7187 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7187 row children.1 children.2) row7187_eq_data branches
set_option maxRecDepth 4096 in
theorem row7193_eq_data : row7193 = Data.row7193 := rfl
def node7193 : Graph := .node 7193 row7193 node7187 Graph.empty
set_option maxRecDepth 4096 in
theorem node7193_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7193 :=
  Budget.node_covers _ _ 7193 row7193 node7187 Graph.empty
    node7187_covers empty_covers (by decide) (by decide) row7193_covers
theorem node7193_eq_data : node7193 = Data.node7193 := by
  have branches : (node7187, Graph.empty) = (Data.node7187, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7187_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7193 row children.1 children.2) row7193_eq_data branches
set_option maxRecDepth 4096 in
theorem row7211_eq_data : row7211 = Data.row7211 := rfl
def node7211 : Graph := .node 7211 row7211 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7211_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7211 :=
  Budget.node_covers _ _ 7211 row7211 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7211_covers
theorem node7211_eq_data : node7211 = Data.node7211 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7211 row children.1 children.2) row7211_eq_data branches
set_option maxRecDepth 4096 in
theorem row7213_eq_data : row7213 = Data.row7213 := rfl
def node7213 : Graph := .node 7213 row7213 node7211 Graph.empty
set_option maxRecDepth 4096 in
theorem node7213_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7213 :=
  Budget.node_covers _ _ 7213 row7213 node7211 Graph.empty
    node7211_covers empty_covers (by decide) (by decide) row7213_covers
theorem node7213_eq_data : node7213 = Data.node7213 := by
  have branches : (node7211, Graph.empty) = (Data.node7211, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7211_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7213 row children.1 children.2) row7213_eq_data branches
set_option maxRecDepth 4096 in
theorem row7207_eq_data : row7207 = Data.row7207 := rfl
def node7207 : Graph := .node 7207 row7207 node7193 node7213
set_option maxRecDepth 4096 in
theorem node7207_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7207 :=
  Budget.node_covers _ _ 7207 row7207 node7193 node7213
    node7193_covers node7213_covers (by decide) (by decide) row7207_covers
theorem node7207_eq_data : node7207 = Data.node7207 := by
  have branches : (node7193, node7213) = (Data.node7193, Data.node7213) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7193_eq_data node7213_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7207 row children.1 children.2) row7207_eq_data branches
set_option maxRecDepth 4096 in
theorem row7177_eq_data : row7177 = Data.row7177 := rfl
def node7177 : Graph := .node 7177 row7177 node7129 node7207
set_option maxRecDepth 4096 in
theorem node7177_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7177 :=
  Budget.node_covers _ _ 7177 row7177 node7129 node7207
    node7129_covers node7207_covers (by decide) (by decide) row7177_covers
theorem node7177_eq_data : node7177 = Data.node7177 := by
  have branches : (node7129, node7207) = (Data.node7129, Data.node7207) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7129_eq_data node7207_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7177 row children.1 children.2) row7177_eq_data branches
set_option maxRecDepth 4096 in
theorem row7109_eq_data : row7109 = Data.row7109 := rfl
def node7109 : Graph := .node 7109 row7109 node7039 node7177
set_option maxRecDepth 4096 in
theorem node7109_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7109 :=
  Budget.node_covers _ _ 7109 row7109 node7039 node7177
    node7039_covers node7177_covers (by decide) (by decide) row7109_covers
theorem node7109_eq_data : node7109 = Data.node7109 := by
  have branches : (node7039, node7177) = (Data.node7039, Data.node7177) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7039_eq_data node7177_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7109 row children.1 children.2) row7109_eq_data branches
set_option maxRecDepth 4096 in
theorem row6991_eq_data : row6991 = Data.row6991 := rfl
def node6991 : Graph := .node 6991 row6991 node6899 node7109
set_option maxRecDepth 4096 in
theorem node6991_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6991 :=
  Budget.node_covers _ _ 6991 row6991 node6899 node7109
    node6899_covers node7109_covers (by decide) (by decide) row6991_covers
theorem node6991_eq_data : node6991 = Data.node6991 := by
  have branches : (node6899, node7109) = (Data.node6899, Data.node7109) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6899_eq_data node7109_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6991 row children.1 children.2) row6991_eq_data branches
set_option maxRecDepth 4096 in
theorem row7229_eq_data : row7229 = Data.row7229 := rfl
def node7229 : Graph := .node 7229 row7229 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7229_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7229 :=
  Budget.node_covers _ _ 7229 row7229 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7229_covers
theorem node7229_eq_data : node7229 = Data.node7229 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7229 row children.1 children.2) row7229_eq_data branches
set_option maxRecDepth 4096 in
theorem row7237_eq_data : row7237 = Data.row7237 := rfl
def node7237 : Graph := .node 7237 row7237 node7229 Graph.empty
set_option maxRecDepth 4096 in
theorem node7237_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7237 :=
  Budget.node_covers _ _ 7237 row7237 node7229 Graph.empty
    node7229_covers empty_covers (by decide) (by decide) row7237_covers
theorem node7237_eq_data : node7237 = Data.node7237 := by
  have branches : (node7229, Graph.empty) = (Data.node7229, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7229_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7237 row children.1 children.2) row7237_eq_data branches
set_option maxRecDepth 4096 in
theorem row7247_eq_data : row7247 = Data.row7247 := rfl
def node7247 : Graph := .node 7247 row7247 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7247_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7247 :=
  Budget.node_covers _ _ 7247 row7247 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7247_covers
theorem node7247_eq_data : node7247 = Data.node7247 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7247 row children.1 children.2) row7247_eq_data branches
set_option maxRecDepth 4096 in
theorem row7253_eq_data : row7253 = Data.row7253 := rfl
def node7253 : Graph := .node 7253 row7253 node7247 Graph.empty
set_option maxRecDepth 4096 in
theorem node7253_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7253 :=
  Budget.node_covers _ _ 7253 row7253 node7247 Graph.empty
    node7247_covers empty_covers (by decide) (by decide) row7253_covers
theorem node7253_eq_data : node7253 = Data.node7253 := by
  have branches : (node7247, Graph.empty) = (Data.node7247, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7247_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7253 row children.1 children.2) row7253_eq_data branches
set_option maxRecDepth 4096 in
theorem row7243_eq_data : row7243 = Data.row7243 := rfl
def node7243 : Graph := .node 7243 row7243 node7237 node7253
set_option maxRecDepth 4096 in
theorem node7243_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7243 :=
  Budget.node_covers _ _ 7243 row7243 node7237 node7253
    node7237_covers node7253_covers (by decide) (by decide) row7243_covers
theorem node7243_eq_data : node7243 = Data.node7243 := by
  have branches : (node7237, node7253) = (Data.node7237, Data.node7253) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7237_eq_data node7253_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7243 row children.1 children.2) row7243_eq_data branches
set_option maxRecDepth 4096 in
theorem row7297_eq_data : row7297 = Data.row7297 := rfl
def node7297 : Graph := .node 7297 row7297 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7297_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7297 :=
  Budget.node_covers _ _ 7297 row7297 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7297_covers
theorem node7297_eq_data : node7297 = Data.node7297 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7297 row children.1 children.2) row7297_eq_data branches
set_option maxRecDepth 4096 in
theorem row7307_eq_data : row7307 = Data.row7307 := rfl
def node7307 : Graph := .node 7307 row7307 node7297 Graph.empty
set_option maxRecDepth 4096 in
theorem node7307_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7307 :=
  Budget.node_covers _ _ 7307 row7307 node7297 Graph.empty
    node7297_covers empty_covers (by decide) (by decide) row7307_covers
theorem node7307_eq_data : node7307 = Data.node7307 := by
  have branches : (node7297, Graph.empty) = (Data.node7297, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7297_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7307 row children.1 children.2) row7307_eq_data branches
set_option maxRecDepth 4096 in
theorem row7321_eq_data : row7321 = Data.row7321 := rfl
def node7321 : Graph := .node 7321 row7321 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7321_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7321 :=
  Budget.node_covers _ _ 7321 row7321 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7321_covers
theorem node7321_eq_data : node7321 = Data.node7321 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7321 row children.1 children.2) row7321_eq_data branches
set_option maxRecDepth 4096 in
theorem row7331_eq_data : row7331 = Data.row7331 := rfl
def node7331 : Graph := .node 7331 row7331 node7321 Graph.empty
set_option maxRecDepth 4096 in
theorem node7331_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7331 :=
  Budget.node_covers _ _ 7331 row7331 node7321 Graph.empty
    node7321_covers empty_covers (by decide) (by decide) row7331_covers
theorem node7331_eq_data : node7331 = Data.node7331 := by
  have branches : (node7321, Graph.empty) = (Data.node7321, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7321_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7331 row children.1 children.2) row7331_eq_data branches
set_option maxRecDepth 4096 in
theorem row7309_eq_data : row7309 = Data.row7309 := rfl
def node7309 : Graph := .node 7309 row7309 node7307 node7331
set_option maxRecDepth 4096 in
theorem node7309_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7309 :=
  Budget.node_covers _ _ 7309 row7309 node7307 node7331
    node7307_covers node7331_covers (by decide) (by decide) row7309_covers
theorem node7309_eq_data : node7309 = Data.node7309 := by
  have branches : (node7307, node7331) = (Data.node7307, Data.node7331) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7307_eq_data node7331_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7309 row children.1 children.2) row7309_eq_data branches
set_option maxRecDepth 4096 in
theorem row7283_eq_data : row7283 = Data.row7283 := rfl
def node7283 : Graph := .node 7283 row7283 node7243 node7309
set_option maxRecDepth 4096 in
theorem node7283_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7283 :=
  Budget.node_covers _ _ 7283 row7283 node7243 node7309
    node7243_covers node7309_covers (by decide) (by decide) row7283_covers
theorem node7283_eq_data : node7283 = Data.node7283 := by
  have branches : (node7243, node7309) = (Data.node7243, Data.node7309) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7243_eq_data node7309_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7283 row children.1 children.2) row7283_eq_data branches
set_option maxRecDepth 4096 in
theorem row7349_eq_data : row7349 = Data.row7349 := rfl
def node7349 : Graph := .node 7349 row7349 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7349_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7349 :=
  Budget.node_covers _ _ 7349 row7349 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7349_covers
theorem node7349_eq_data : node7349 = Data.node7349 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7349 row children.1 children.2) row7349_eq_data branches
set_option maxRecDepth 4096 in
theorem row7351_eq_data : row7351 = Data.row7351 := rfl
def node7351 : Graph := .node 7351 row7351 node7349 Graph.empty
set_option maxRecDepth 4096 in
theorem node7351_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7351 :=
  Budget.node_covers _ _ 7351 row7351 node7349 Graph.empty
    node7349_covers empty_covers (by decide) (by decide) row7351_covers
theorem node7351_eq_data : node7351 = Data.node7351 := by
  have branches : (node7349, Graph.empty) = (Data.node7349, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7349_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7351 row children.1 children.2) row7351_eq_data branches
set_option maxRecDepth 4096 in
theorem row7393_eq_data : row7393 = Data.row7393 := rfl
def node7393 : Graph := .node 7393 row7393 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7393_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7393 :=
  Budget.node_covers _ _ 7393 row7393 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7393_covers
theorem node7393_eq_data : node7393 = Data.node7393 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7393 row children.1 children.2) row7393_eq_data branches
set_option maxRecDepth 4096 in
theorem row7411_eq_data : row7411 = Data.row7411 := rfl
def node7411 : Graph := .node 7411 row7411 node7393 Graph.empty
set_option maxRecDepth 4096 in
theorem node7411_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7411 :=
  Budget.node_covers _ _ 7411 row7411 node7393 Graph.empty
    node7393_covers empty_covers (by decide) (by decide) row7411_covers
theorem node7411_eq_data : node7411 = Data.node7411 := by
  have branches : (node7393, Graph.empty) = (Data.node7393, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7393_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7411 row children.1 children.2) row7411_eq_data branches
set_option maxRecDepth 4096 in
theorem row7369_eq_data : row7369 = Data.row7369 := rfl
def node7369 : Graph := .node 7369 row7369 node7351 node7411
set_option maxRecDepth 4096 in
theorem node7369_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7369 :=
  Budget.node_covers _ _ 7369 row7369 node7351 node7411
    node7351_covers node7411_covers (by decide) (by decide) row7369_covers
theorem node7369_eq_data : node7369 = Data.node7369 := by
  have branches : (node7351, node7411) = (Data.node7351, Data.node7411) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7351_eq_data node7411_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7369 row children.1 children.2) row7369_eq_data branches
set_option maxRecDepth 4096 in
theorem row7433_eq_data : row7433 = Data.row7433 := rfl
def node7433 : Graph := .node 7433 row7433 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7433 :=
  Budget.node_covers _ _ 7433 row7433 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7433_covers
theorem node7433_eq_data : node7433 = Data.node7433 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7433 row children.1 children.2) row7433_eq_data branches
set_option maxRecDepth 4096 in
theorem row7451_eq_data : row7451 = Data.row7451 := rfl
def node7451 : Graph := .node 7451 row7451 node7433 Graph.empty
set_option maxRecDepth 4096 in
theorem node7451_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7451 :=
  Budget.node_covers _ _ 7451 row7451 node7433 Graph.empty
    node7433_covers empty_covers (by decide) (by decide) row7451_covers
theorem node7451_eq_data : node7451 = Data.node7451 := by
  have branches : (node7433, Graph.empty) = (Data.node7433, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7433_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7451 row children.1 children.2) row7451_eq_data branches
set_option maxRecDepth 4096 in
theorem row7459_eq_data : row7459 = Data.row7459 := rfl
def node7459 : Graph := .node 7459 row7459 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7459_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7459 :=
  Budget.node_covers _ _ 7459 row7459 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7459_covers
theorem node7459_eq_data : node7459 = Data.node7459 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7459 row children.1 children.2) row7459_eq_data branches
set_option maxRecDepth 4096 in
theorem row7477_eq_data : row7477 = Data.row7477 := rfl
def node7477 : Graph := .node 7477 row7477 node7459 Graph.empty
set_option maxRecDepth 4096 in
theorem node7477_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7477 :=
  Budget.node_covers _ _ 7477 row7477 node7459 Graph.empty
    node7459_covers empty_covers (by decide) (by decide) row7477_covers
theorem node7477_eq_data : node7477 = Data.node7477 := by
  have branches : (node7459, Graph.empty) = (Data.node7459, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7459_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7477 row children.1 children.2) row7477_eq_data branches
set_option maxRecDepth 4096 in
theorem row7457_eq_data : row7457 = Data.row7457 := rfl
def node7457 : Graph := .node 7457 row7457 node7451 node7477
set_option maxRecDepth 4096 in
theorem node7457_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7457 :=
  Budget.node_covers _ _ 7457 row7457 node7451 node7477
    node7451_covers node7477_covers (by decide) (by decide) row7457_covers
theorem node7457_eq_data : node7457 = Data.node7457 := by
  have branches : (node7451, node7477) = (Data.node7451, Data.node7477) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7451_eq_data node7477_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7457 row children.1 children.2) row7457_eq_data branches
set_option maxRecDepth 4096 in
theorem row7417_eq_data : row7417 = Data.row7417 := rfl
def node7417 : Graph := .node 7417 row7417 node7369 node7457
set_option maxRecDepth 4096 in
theorem node7417_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7417 :=
  Budget.node_covers _ _ 7417 row7417 node7369 node7457
    node7369_covers node7457_covers (by decide) (by decide) row7417_covers
theorem node7417_eq_data : node7417 = Data.node7417 := by
  have branches : (node7369, node7457) = (Data.node7369, Data.node7457) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7369_eq_data node7457_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7417 row children.1 children.2) row7417_eq_data branches
set_option maxRecDepth 4096 in
theorem row7333_eq_data : row7333 = Data.row7333 := rfl
def node7333 : Graph := .node 7333 row7333 node7283 node7417
set_option maxRecDepth 4096 in
theorem node7333_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7333 :=
  Budget.node_covers _ _ 7333 row7333 node7283 node7417
    node7283_covers node7417_covers (by decide) (by decide) row7333_covers
theorem node7333_eq_data : node7333 = Data.node7333 := by
  have branches : (node7283, node7417) = (Data.node7283, Data.node7417) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7283_eq_data node7417_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7333 row children.1 children.2) row7333_eq_data branches
set_option maxRecDepth 4096 in
theorem row7487_eq_data : row7487 = Data.row7487 := rfl
def node7487 : Graph := .node 7487 row7487 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7487_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7487 :=
  Budget.node_covers _ _ 7487 row7487 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7487_covers
theorem node7487_eq_data : node7487 = Data.node7487 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7487 row children.1 children.2) row7487_eq_data branches
set_option maxRecDepth 4096 in
theorem row7489_eq_data : row7489 = Data.row7489 := rfl
def node7489 : Graph := .node 7489 row7489 node7487 Graph.empty
set_option maxRecDepth 4096 in
theorem node7489_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7489 :=
  Budget.node_covers _ _ 7489 row7489 node7487 Graph.empty
    node7487_covers empty_covers (by decide) (by decide) row7489_covers
theorem node7489_eq_data : node7489 = Data.node7489 := by
  have branches : (node7487, Graph.empty) = (Data.node7487, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7487_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7489 row children.1 children.2) row7489_eq_data branches
set_option maxRecDepth 4096 in
theorem row7507_eq_data : row7507 = Data.row7507 := rfl
def node7507 : Graph := .node 7507 row7507 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7507_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7507 :=
  Budget.node_covers _ _ 7507 row7507 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7507_covers
theorem node7507_eq_data : node7507 = Data.node7507 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7507 row children.1 children.2) row7507_eq_data branches
set_option maxRecDepth 4096 in
theorem row7517_eq_data : row7517 = Data.row7517 := rfl
def node7517 : Graph := .node 7517 row7517 node7507 Graph.empty
set_option maxRecDepth 4096 in
theorem node7517_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7517 :=
  Budget.node_covers _ _ 7517 row7517 node7507 Graph.empty
    node7507_covers empty_covers (by decide) (by decide) row7517_covers
theorem node7517_eq_data : node7517 = Data.node7517 := by
  have branches : (node7507, Graph.empty) = (Data.node7507, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7507_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7517 row children.1 children.2) row7517_eq_data branches
set_option maxRecDepth 4096 in
theorem row7499_eq_data : row7499 = Data.row7499 := rfl
def node7499 : Graph := .node 7499 row7499 node7489 node7517
set_option maxRecDepth 4096 in
theorem node7499_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7499 :=
  Budget.node_covers _ _ 7499 row7499 node7489 node7517
    node7489_covers node7517_covers (by decide) (by decide) row7499_covers
theorem node7499_eq_data : node7499 = Data.node7499 := by
  have branches : (node7489, node7517) = (Data.node7489, Data.node7517) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7489_eq_data node7517_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7499 row children.1 children.2) row7499_eq_data branches
set_option maxRecDepth 4096 in
theorem row7529_eq_data : row7529 = Data.row7529 := rfl
def node7529 : Graph := .node 7529 row7529 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7529_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7529 :=
  Budget.node_covers _ _ 7529 row7529 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7529_covers
theorem node7529_eq_data : node7529 = Data.node7529 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7529 row children.1 children.2) row7529_eq_data branches
set_option maxRecDepth 4096 in
theorem row7537_eq_data : row7537 = Data.row7537 := rfl
def node7537 : Graph := .node 7537 row7537 node7529 Graph.empty
set_option maxRecDepth 4096 in
theorem node7537_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7537 :=
  Budget.node_covers _ _ 7537 row7537 node7529 Graph.empty
    node7529_covers empty_covers (by decide) (by decide) row7537_covers
theorem node7537_eq_data : node7537 = Data.node7537 := by
  have branches : (node7529, Graph.empty) = (Data.node7529, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7529_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7537 row children.1 children.2) row7537_eq_data branches
set_option maxRecDepth 4096 in
theorem row7547_eq_data : row7547 = Data.row7547 := rfl
def node7547 : Graph := .node 7547 row7547 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7547 :=
  Budget.node_covers _ _ 7547 row7547 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7547_covers
theorem node7547_eq_data : node7547 = Data.node7547 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7547 row children.1 children.2) row7547_eq_data branches
set_option maxRecDepth 4096 in
theorem row7549_eq_data : row7549 = Data.row7549 := rfl
def node7549 : Graph := .node 7549 row7549 node7547 Graph.empty
set_option maxRecDepth 4096 in
theorem node7549_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7549 :=
  Budget.node_covers _ _ 7549 row7549 node7547 Graph.empty
    node7547_covers empty_covers (by decide) (by decide) row7549_covers
theorem node7549_eq_data : node7549 = Data.node7549 := by
  have branches : (node7547, Graph.empty) = (Data.node7547, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7547_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7549 row children.1 children.2) row7549_eq_data branches
set_option maxRecDepth 4096 in
theorem row7541_eq_data : row7541 = Data.row7541 := rfl
def node7541 : Graph := .node 7541 row7541 node7537 node7549
set_option maxRecDepth 4096 in
theorem node7541_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7541 :=
  Budget.node_covers _ _ 7541 row7541 node7537 node7549
    node7537_covers node7549_covers (by decide) (by decide) row7541_covers
theorem node7541_eq_data : node7541 = Data.node7541 := by
  have branches : (node7537, node7549) = (Data.node7537, Data.node7549) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7537_eq_data node7549_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7541 row children.1 children.2) row7541_eq_data branches
set_option maxRecDepth 4096 in
theorem row7523_eq_data : row7523 = Data.row7523 := rfl
def node7523 : Graph := .node 7523 row7523 node7499 node7541
set_option maxRecDepth 4096 in
theorem node7523_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7523 :=
  Budget.node_covers _ _ 7523 row7523 node7499 node7541
    node7499_covers node7541_covers (by decide) (by decide) row7523_covers
theorem node7523_eq_data : node7523 = Data.node7523 := by
  have branches : (node7499, node7541) = (Data.node7499, Data.node7541) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7499_eq_data node7541_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7523 row children.1 children.2) row7523_eq_data branches
set_option maxRecDepth 4096 in
theorem row7561_eq_data : row7561 = Data.row7561 := rfl
def node7561 : Graph := .node 7561 row7561 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7561_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7561 :=
  Budget.node_covers _ _ 7561 row7561 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7561_covers
theorem node7561_eq_data : node7561 = Data.node7561 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7561 row children.1 children.2) row7561_eq_data branches
set_option maxRecDepth 4096 in
theorem row7573_eq_data : row7573 = Data.row7573 := rfl
def node7573 : Graph := .node 7573 row7573 node7561 Graph.empty
set_option maxRecDepth 4096 in
theorem node7573_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7573 :=
  Budget.node_covers _ _ 7573 row7573 node7561 Graph.empty
    node7561_covers empty_covers (by decide) (by decide) row7573_covers
theorem node7573_eq_data : node7573 = Data.node7573 := by
  have branches : (node7561, Graph.empty) = (Data.node7561, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7561_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7573 row children.1 children.2) row7573_eq_data branches
set_option maxRecDepth 4096 in
theorem row7583_eq_data : row7583 = Data.row7583 := rfl
def node7583 : Graph := .node 7583 row7583 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7583_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7583 :=
  Budget.node_covers _ _ 7583 row7583 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7583_covers
theorem node7583_eq_data : node7583 = Data.node7583 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7583 row children.1 children.2) row7583_eq_data branches
set_option maxRecDepth 4096 in
theorem row7589_eq_data : row7589 = Data.row7589 := rfl
def node7589 : Graph := .node 7589 row7589 node7583 Graph.empty
set_option maxRecDepth 4096 in
theorem node7589_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7589 :=
  Budget.node_covers _ _ 7589 row7589 node7583 Graph.empty
    node7583_covers empty_covers (by decide) (by decide) row7589_covers
theorem node7589_eq_data : node7589 = Data.node7589 := by
  have branches : (node7583, Graph.empty) = (Data.node7583, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7583_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7589 row children.1 children.2) row7589_eq_data branches
set_option maxRecDepth 4096 in
theorem row7577_eq_data : row7577 = Data.row7577 := rfl
def node7577 : Graph := .node 7577 row7577 node7573 node7589
set_option maxRecDepth 4096 in
theorem node7577_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7577 :=
  Budget.node_covers _ _ 7577 row7577 node7573 node7589
    node7573_covers node7589_covers (by decide) (by decide) row7577_covers
theorem node7577_eq_data : node7577 = Data.node7577 := by
  have branches : (node7573, node7589) = (Data.node7573, Data.node7589) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7573_eq_data node7589_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7577 row children.1 children.2) row7577_eq_data branches
set_option maxRecDepth 4096 in
theorem row7603_eq_data : row7603 = Data.row7603 := rfl
def node7603 : Graph := .node 7603 row7603 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7603_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7603 :=
  Budget.node_covers _ _ 7603 row7603 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7603_covers
theorem node7603_eq_data : node7603 = Data.node7603 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7603 row children.1 children.2) row7603_eq_data branches
set_option maxRecDepth 4096 in
theorem row7607_eq_data : row7607 = Data.row7607 := rfl
def node7607 : Graph := .node 7607 row7607 node7603 Graph.empty
set_option maxRecDepth 4096 in
theorem node7607_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7607 :=
  Budget.node_covers _ _ 7607 row7607 node7603 Graph.empty
    node7603_covers empty_covers (by decide) (by decide) row7607_covers
theorem node7607_eq_data : node7607 = Data.node7607 := by
  have branches : (node7603, Graph.empty) = (Data.node7603, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7603_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7607 row children.1 children.2) row7607_eq_data branches
set_option maxRecDepth 4096 in
theorem row7639_eq_data : row7639 = Data.row7639 := rfl
def node7639 : Graph := .node 7639 row7639 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node7639_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7639 :=
  Budget.node_covers _ _ 7639 row7639 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row7639_covers
theorem node7639_eq_data : node7639 = Data.node7639 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7639 row children.1 children.2) row7639_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
