import ProjectEulerSolutions.Lib.PrimePairLarge.Graph023
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row12983_eq_data : row12983 = Data.row12983 := rfl
def node12983 : Graph := .node 12983 row12983 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12983_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12983 :=
  Budget.node_covers _ _ 12983 row12983 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12983_covers
theorem node12983_eq_data : node12983 = Data.node12983 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12983 row children.1 children.2) row12983_eq_data branches
set_option maxRecDepth 4096 in
theorem row13001_eq_data : row13001 = Data.row13001 := rfl
def node13001 : Graph := .node 13001 row13001 node12983 Graph.empty
set_option maxRecDepth 4096 in
theorem node13001_covers : Budget.Covers PrimePairVertices.largeTree 26034 node13001 :=
  Budget.node_covers _ _ 13001 row13001 node12983 Graph.empty
    node12983_covers empty_covers (by decide) (by decide) row13001_covers
theorem node13001_eq_data : node13001 = Data.node13001 := by
  have branches : (node12983, Graph.empty) = (Data.node12983, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12983_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 13001 row children.1 children.2) row13001_eq_data branches
set_option maxRecDepth 4096 in
theorem row13007_eq_data : row13007 = Data.row13007 := rfl
def node13007 : Graph := .node 13007 row13007 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node13007_covers : Budget.Covers PrimePairVertices.largeTree 26034 node13007 :=
  Budget.node_covers _ _ 13007 row13007 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row13007_covers
theorem node13007_eq_data : node13007 = Data.node13007 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 13007 row children.1 children.2) row13007_eq_data branches
set_option maxRecDepth 4096 in
theorem row13009_eq_data : row13009 = Data.row13009 := rfl
def node13009 : Graph := .node 13009 row13009 node13007 Graph.empty
set_option maxRecDepth 4096 in
theorem node13009_covers : Budget.Covers PrimePairVertices.largeTree 26034 node13009 :=
  Budget.node_covers _ _ 13009 row13009 node13007 Graph.empty
    node13007_covers empty_covers (by decide) (by decide) row13009_covers
theorem node13009_eq_data : node13009 = Data.node13009 := by
  have branches : (node13007, Graph.empty) = (Data.node13007, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node13007_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 13009 row children.1 children.2) row13009_eq_data branches
set_option maxRecDepth 4096 in
theorem row13003_eq_data : row13003 = Data.row13003 := rfl
def node13003 : Graph := .node 13003 row13003 node13001 node13009
set_option maxRecDepth 4096 in
theorem node13003_covers : Budget.Covers PrimePairVertices.largeTree 26034 node13003 :=
  Budget.node_covers _ _ 13003 row13003 node13001 node13009
    node13001_covers node13009_covers (by decide) (by decide) row13003_covers
theorem node13003_eq_data : node13003 = Data.node13003 := by
  have branches : (node13001, node13009) = (Data.node13001, Data.node13009) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node13001_eq_data node13009_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 13003 row children.1 children.2) row13003_eq_data branches
set_option maxRecDepth 4096 in
theorem row12979_eq_data : row12979 = Data.row12979 := rfl
def node12979 : Graph := .node 12979 row12979 node12959 node13003
set_option maxRecDepth 4096 in
theorem node12979_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12979 :=
  Budget.node_covers _ _ 12979 row12979 node12959 node13003
    node12959_covers node13003_covers (by decide) (by decide) row12979_covers
theorem node12979_eq_data : node12979 = Data.node12979 := by
  have branches : (node12959, node13003) = (Data.node12959, Data.node13003) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12959_eq_data node13003_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12979 row children.1 children.2) row12979_eq_data branches
set_option maxRecDepth 4096 in
theorem row12923_eq_data : row12923 = Data.row12923 := rfl
def node12923 : Graph := .node 12923 row12923 node12893 node12979
set_option maxRecDepth 4096 in
theorem node12923_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12923 :=
  Budget.node_covers _ _ 12923 row12923 node12893 node12979
    node12893_covers node12979_covers (by decide) (by decide) row12923_covers
theorem node12923_eq_data : node12923 = Data.node12923 := by
  have branches : (node12893, node12979) = (Data.node12893, Data.node12979) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12893_eq_data node12979_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12923 row children.1 children.2) row12923_eq_data branches
set_option maxRecDepth 4096 in
theorem row12821_eq_data : row12821 = Data.row12821 := rfl
def node12821 : Graph := .node 12821 row12821 node12697 node12923
set_option maxRecDepth 4096 in
theorem node12821_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12821 :=
  Budget.node_covers _ _ 12821 row12821 node12697 node12923
    node12697_covers node12923_covers (by decide) (by decide) row12821_covers
theorem node12821_eq_data : node12821 = Data.node12821 := by
  have branches : (node12697, node12923) = (Data.node12697, Data.node12923) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12697_eq_data node12923_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12821 row children.1 children.2) row12821_eq_data branches
set_option maxRecDepth 4096 in
theorem row12589_eq_data : row12589 = Data.row12589 := rfl
def node12589 : Graph := .node 12589 row12589 node12401 node12821
set_option maxRecDepth 4096 in
theorem node12589_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12589 :=
  Budget.node_covers _ _ 12589 row12589 node12401 node12821
    node12401_covers node12821_covers (by decide) (by decide) row12589_covers
theorem node12589_eq_data : node12589 = Data.node12589 := by
  have branches : (node12401, node12821) = (Data.node12401, Data.node12821) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12401_eq_data node12821_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12589 row children.1 children.2) row12589_eq_data branches
set_option maxRecDepth 4096 in
theorem row12161_eq_data : row12161 = Data.row12161 := rfl
def node12161 : Graph := .node 12161 row12161 node11731 node12589
set_option maxRecDepth 4096 in
theorem node12161_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12161 :=
  Budget.node_covers _ _ 12161 row12161 node11731 node12589
    node11731_covers node12589_covers (by decide) (by decide) row12161_covers
theorem node12161_eq_data : node12161 = Data.node12161 := by
  have branches : (node11731, node12589) = (Data.node11731, Data.node12589) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11731_eq_data node12589_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12161 row children.1 children.2) row12161_eq_data branches
set_option maxRecDepth 4096 in
theorem row11243_eq_data : row11243 = Data.row11243 := rfl
def node11243 : Graph := .node 11243 row11243 node10289 node12161
set_option maxRecDepth 4096 in
theorem node11243_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11243 :=
  Budget.node_covers _ _ 11243 row11243 node10289 node12161
    node10289_covers node12161_covers (by decide) (by decide) row11243_covers
theorem node11243_eq_data : node11243 = Data.node11243 := by
  have branches : (node10289, node12161) = (Data.node10289, Data.node12161) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10289_eq_data node12161_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11243 row children.1 children.2) row11243_eq_data branches
set_option maxRecDepth 4096 in
theorem row9419_eq_data : row9419 = Data.row9419 := rfl
def node9419 : Graph := .node 9419 row9419 node7649 node11243
set_option maxRecDepth 4096 in
theorem node9419_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9419 :=
  Budget.node_covers _ _ 9419 row9419 node7649 node11243
    node7649_covers node11243_covers (by decide) (by decide) row9419_covers
theorem node9419_eq_data : node9419 = Data.node9419 := by
  have branches : (node7649, node11243) = (Data.node7649, Data.node11243) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7649_eq_data node11243_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9419 row children.1 children.2) row9419_eq_data branches
set_option maxRecDepth 4096 in
theorem row5903_eq_data : row5903 = Data.row5903 := rfl
def node5903 : Graph := .node 5903 row5903 node2687 node9419
set_option maxRecDepth 4096 in
theorem node5903_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5903 :=
  Budget.node_covers _ _ 5903 row5903 node2687 node9419
    node2687_covers node9419_covers (by decide) (by decide) row5903_covers
theorem node5903_eq_data : node5903 = Data.node5903 := by
  have branches : (node2687, node9419) = (Data.node2687, Data.node9419) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2687_eq_data node9419_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5903 row children.1 children.2) row5903_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
