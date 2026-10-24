import ProjectEulerSolutions.Lib.PrimePairLarge.Graph016
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row8783_eq_data : row8783 = Data.row8783 := rfl
def node8783 : Graph := .node 8783 row8783 node8779 Graph.empty
set_option maxRecDepth 4096 in
theorem node8783_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8783 :=
  Budget.node_covers _ _ 8783 row8783 node8779 Graph.empty
    node8779_covers empty_covers (by decide) (by decide) row8783_covers
theorem node8783_eq_data : node8783 = Data.node8783 := by
  have branches : (node8779, Graph.empty) = (Data.node8779, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8779_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8783 row children.1 children.2) row8783_eq_data branches
set_option maxRecDepth 4096 in
theorem row8807_eq_data : row8807 = Data.row8807 := rfl
def node8807 : Graph := .node 8807 row8807 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8807_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8807 :=
  Budget.node_covers _ _ 8807 row8807 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8807_covers
theorem node8807_eq_data : node8807 = Data.node8807 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8807 row children.1 children.2) row8807_eq_data branches
set_option maxRecDepth 4096 in
theorem row8819_eq_data : row8819 = Data.row8819 := rfl
def node8819 : Graph := .node 8819 row8819 node8807 Graph.empty
set_option maxRecDepth 4096 in
theorem node8819_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8819 :=
  Budget.node_covers _ _ 8819 row8819 node8807 Graph.empty
    node8807_covers empty_covers (by decide) (by decide) row8819_covers
theorem node8819_eq_data : node8819 = Data.node8819 := by
  have branches : (node8807, Graph.empty) = (Data.node8807, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8807_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8819 row children.1 children.2) row8819_eq_data branches
set_option maxRecDepth 4096 in
theorem row8803_eq_data : row8803 = Data.row8803 := rfl
def node8803 : Graph := .node 8803 row8803 node8783 node8819
set_option maxRecDepth 4096 in
theorem node8803_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8803 :=
  Budget.node_covers _ _ 8803 row8803 node8783 node8819
    node8783_covers node8819_covers (by decide) (by decide) row8803_covers
theorem node8803_eq_data : node8803 = Data.node8803 := by
  have branches : (node8783, node8819) = (Data.node8783, Data.node8819) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8783_eq_data node8819_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8803 row children.1 children.2) row8803_eq_data branches
set_option maxRecDepth 4096 in
theorem row8831_eq_data : row8831 = Data.row8831 := rfl
def node8831 : Graph := .node 8831 row8831 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8831_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8831 :=
  Budget.node_covers _ _ 8831 row8831 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8831_covers
theorem node8831_eq_data : node8831 = Data.node8831 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8831 row children.1 children.2) row8831_eq_data branches
set_option maxRecDepth 4096 in
theorem row8837_eq_data : row8837 = Data.row8837 := rfl
def node8837 : Graph := .node 8837 row8837 node8831 Graph.empty
set_option maxRecDepth 4096 in
theorem node8837_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8837 :=
  Budget.node_covers _ _ 8837 row8837 node8831 Graph.empty
    node8831_covers empty_covers (by decide) (by decide) row8837_covers
theorem node8837_eq_data : node8837 = Data.node8837 := by
  have branches : (node8831, Graph.empty) = (Data.node8831, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8831_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8837 row children.1 children.2) row8837_eq_data branches
set_option maxRecDepth 4096 in
theorem row8849_eq_data : row8849 = Data.row8849 := rfl
def node8849 : Graph := .node 8849 row8849 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8849_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8849 :=
  Budget.node_covers _ _ 8849 row8849 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8849_covers
theorem node8849_eq_data : node8849 = Data.node8849 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8849 row children.1 children.2) row8849_eq_data branches
set_option maxRecDepth 4096 in
theorem row8861_eq_data : row8861 = Data.row8861 := rfl
def node8861 : Graph := .node 8861 row8861 node8849 Graph.empty
set_option maxRecDepth 4096 in
theorem node8861_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8861 :=
  Budget.node_covers _ _ 8861 row8861 node8849 Graph.empty
    node8849_covers empty_covers (by decide) (by decide) row8861_covers
theorem node8861_eq_data : node8861 = Data.node8861 := by
  have branches : (node8849, Graph.empty) = (Data.node8849, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8849_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8861 row children.1 children.2) row8861_eq_data branches
set_option maxRecDepth 4096 in
theorem row8839_eq_data : row8839 = Data.row8839 := rfl
def node8839 : Graph := .node 8839 row8839 node8837 node8861
set_option maxRecDepth 4096 in
theorem node8839_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8839 :=
  Budget.node_covers _ _ 8839 row8839 node8837 node8861
    node8837_covers node8861_covers (by decide) (by decide) row8839_covers
theorem node8839_eq_data : node8839 = Data.node8839 := by
  have branches : (node8837, node8861) = (Data.node8837, Data.node8861) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8837_eq_data node8861_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8839 row children.1 children.2) row8839_eq_data branches
set_option maxRecDepth 4096 in
theorem row8821_eq_data : row8821 = Data.row8821 := rfl
def node8821 : Graph := .node 8821 row8821 node8803 node8839
set_option maxRecDepth 4096 in
theorem node8821_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8821 :=
  Budget.node_covers _ _ 8821 row8821 node8803 node8839
    node8803_covers node8839_covers (by decide) (by decide) row8821_covers
theorem node8821_eq_data : node8821 = Data.node8821 := by
  have branches : (node8803, node8839) = (Data.node8803, Data.node8839) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8803_eq_data node8839_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8821 row children.1 children.2) row8821_eq_data branches
set_option maxRecDepth 4096 in
theorem row8867_eq_data : row8867 = Data.row8867 := rfl
def node8867 : Graph := .node 8867 row8867 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8867_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8867 :=
  Budget.node_covers _ _ 8867 row8867 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8867_covers
theorem node8867_eq_data : node8867 = Data.node8867 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8867 row children.1 children.2) row8867_eq_data branches
set_option maxRecDepth 4096 in
theorem row8887_eq_data : row8887 = Data.row8887 := rfl
def node8887 : Graph := .node 8887 row8887 node8867 Graph.empty
set_option maxRecDepth 4096 in
theorem node8887_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8887 :=
  Budget.node_covers _ _ 8887 row8887 node8867 Graph.empty
    node8867_covers empty_covers (by decide) (by decide) row8887_covers
theorem node8887_eq_data : node8887 = Data.node8887 := by
  have branches : (node8867, Graph.empty) = (Data.node8867, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8867_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8887 row children.1 children.2) row8887_eq_data branches
set_option maxRecDepth 4096 in
theorem row8923_eq_data : row8923 = Data.row8923 := rfl
def node8923 : Graph := .node 8923 row8923 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8923_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8923 :=
  Budget.node_covers _ _ 8923 row8923 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8923_covers
theorem node8923_eq_data : node8923 = Data.node8923 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8923 row children.1 children.2) row8923_eq_data branches
set_option maxRecDepth 4096 in
theorem row8929_eq_data : row8929 = Data.row8929 := rfl
def node8929 : Graph := .node 8929 row8929 node8923 Graph.empty
set_option maxRecDepth 4096 in
theorem node8929_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8929 :=
  Budget.node_covers _ _ 8929 row8929 node8923 Graph.empty
    node8923_covers empty_covers (by decide) (by decide) row8929_covers
theorem node8929_eq_data : node8929 = Data.node8929 := by
  have branches : (node8923, Graph.empty) = (Data.node8923, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8923_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8929 row children.1 children.2) row8929_eq_data branches
set_option maxRecDepth 4096 in
theorem row8893_eq_data : row8893 = Data.row8893 := rfl
def node8893 : Graph := .node 8893 row8893 node8887 node8929
set_option maxRecDepth 4096 in
theorem node8893_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8893 :=
  Budget.node_covers _ _ 8893 row8893 node8887 node8929
    node8887_covers node8929_covers (by decide) (by decide) row8893_covers
theorem node8893_eq_data : node8893 = Data.node8893 := by
  have branches : (node8887, node8929) = (Data.node8887, Data.node8929) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8887_eq_data node8929_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8893 row children.1 children.2) row8893_eq_data branches
set_option maxRecDepth 4096 in
theorem row8941_eq_data : row8941 = Data.row8941 := rfl
def node8941 : Graph := .node 8941 row8941 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8941_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8941 :=
  Budget.node_covers _ _ 8941 row8941 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8941_covers
theorem node8941_eq_data : node8941 = Data.node8941 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8941 row children.1 children.2) row8941_eq_data branches
set_option maxRecDepth 4096 in
theorem row8951_eq_data : row8951 = Data.row8951 := rfl
def node8951 : Graph := .node 8951 row8951 node8941 Graph.empty
set_option maxRecDepth 4096 in
theorem node8951_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8951 :=
  Budget.node_covers _ _ 8951 row8951 node8941 Graph.empty
    node8941_covers empty_covers (by decide) (by decide) row8951_covers
theorem node8951_eq_data : node8951 = Data.node8951 := by
  have branches : (node8941, Graph.empty) = (Data.node8941, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8941_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8951 row children.1 children.2) row8951_eq_data branches
set_option maxRecDepth 4096 in
theorem row8969_eq_data : row8969 = Data.row8969 := rfl
def node8969 : Graph := .node 8969 row8969 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8969_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8969 :=
  Budget.node_covers _ _ 8969 row8969 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8969_covers
theorem node8969_eq_data : node8969 = Data.node8969 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8969 row children.1 children.2) row8969_eq_data branches
set_option maxRecDepth 4096 in
theorem row8971_eq_data : row8971 = Data.row8971 := rfl
def node8971 : Graph := .node 8971 row8971 node8969 Graph.empty
set_option maxRecDepth 4096 in
theorem node8971_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8971 :=
  Budget.node_covers _ _ 8971 row8971 node8969 Graph.empty
    node8969_covers empty_covers (by decide) (by decide) row8971_covers
theorem node8971_eq_data : node8971 = Data.node8971 := by
  have branches : (node8969, Graph.empty) = (Data.node8969, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8969_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8971 row children.1 children.2) row8971_eq_data branches
set_option maxRecDepth 4096 in
theorem row8963_eq_data : row8963 = Data.row8963 := rfl
def node8963 : Graph := .node 8963 row8963 node8951 node8971
set_option maxRecDepth 4096 in
theorem node8963_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8963 :=
  Budget.node_covers _ _ 8963 row8963 node8951 node8971
    node8951_covers node8971_covers (by decide) (by decide) row8963_covers
theorem node8963_eq_data : node8963 = Data.node8963 := by
  have branches : (node8951, node8971) = (Data.node8951, Data.node8971) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8951_eq_data node8971_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8963 row children.1 children.2) row8963_eq_data branches
set_option maxRecDepth 4096 in
theorem row8933_eq_data : row8933 = Data.row8933 := rfl
def node8933 : Graph := .node 8933 row8933 node8893 node8963
set_option maxRecDepth 4096 in
theorem node8933_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8933 :=
  Budget.node_covers _ _ 8933 row8933 node8893 node8963
    node8893_covers node8963_covers (by decide) (by decide) row8933_covers
theorem node8933_eq_data : node8933 = Data.node8933 := by
  have branches : (node8893, node8963) = (Data.node8893, Data.node8963) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8893_eq_data node8963_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8933 row children.1 children.2) row8933_eq_data branches
set_option maxRecDepth 4096 in
theorem row8863_eq_data : row8863 = Data.row8863 := rfl
def node8863 : Graph := .node 8863 row8863 node8821 node8933
set_option maxRecDepth 4096 in
theorem node8863_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8863 :=
  Budget.node_covers _ _ 8863 row8863 node8821 node8933
    node8821_covers node8933_covers (by decide) (by decide) row8863_covers
theorem node8863_eq_data : node8863 = Data.node8863 := by
  have branches : (node8821, node8933) = (Data.node8821, Data.node8933) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8821_eq_data node8933_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8863 row children.1 children.2) row8863_eq_data branches
set_option maxRecDepth 4096 in
theorem row8761_eq_data : row8761 = Data.row8761 := rfl
def node8761 : Graph := .node 8761 row8761 node8681 node8863
set_option maxRecDepth 4096 in
theorem node8761_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8761 :=
  Budget.node_covers _ _ 8761 row8761 node8681 node8863
    node8681_covers node8863_covers (by decide) (by decide) row8761_covers
theorem node8761_eq_data : node8761 = Data.node8761 := by
  have branches : (node8681, node8863) = (Data.node8681, Data.node8863) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8681_eq_data node8863_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8761 row children.1 children.2) row8761_eq_data branches
set_option maxRecDepth 4096 in
theorem row9001_eq_data : row9001 = Data.row9001 := rfl
def node9001 : Graph := .node 9001 row9001 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9001_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9001 :=
  Budget.node_covers _ _ 9001 row9001 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9001_covers
theorem node9001_eq_data : node9001 = Data.node9001 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9001 row children.1 children.2) row9001_eq_data branches
set_option maxRecDepth 4096 in
theorem row9007_eq_data : row9007 = Data.row9007 := rfl
def node9007 : Graph := .node 9007 row9007 node9001 Graph.empty
set_option maxRecDepth 4096 in
theorem node9007_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9007 :=
  Budget.node_covers _ _ 9007 row9007 node9001 Graph.empty
    node9001_covers empty_covers (by decide) (by decide) row9007_covers
theorem node9007_eq_data : node9007 = Data.node9007 := by
  have branches : (node9001, Graph.empty) = (Data.node9001, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9001_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9007 row children.1 children.2) row9007_eq_data branches
set_option maxRecDepth 4096 in
theorem row9013_eq_data : row9013 = Data.row9013 := rfl
def node9013 : Graph := .node 9013 row9013 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9013_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9013 :=
  Budget.node_covers _ _ 9013 row9013 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9013_covers
theorem node9013_eq_data : node9013 = Data.node9013 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9013 row children.1 children.2) row9013_eq_data branches
set_option maxRecDepth 4096 in
theorem row9029_eq_data : row9029 = Data.row9029 := rfl
def node9029 : Graph := .node 9029 row9029 node9013 Graph.empty
set_option maxRecDepth 4096 in
theorem node9029_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9029 :=
  Budget.node_covers _ _ 9029 row9029 node9013 Graph.empty
    node9013_covers empty_covers (by decide) (by decide) row9029_covers
theorem node9029_eq_data : node9029 = Data.node9029 := by
  have branches : (node9013, Graph.empty) = (Data.node9013, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9013_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9029 row children.1 children.2) row9029_eq_data branches
set_option maxRecDepth 4096 in
theorem row9011_eq_data : row9011 = Data.row9011 := rfl
def node9011 : Graph := .node 9011 row9011 node9007 node9029
set_option maxRecDepth 4096 in
theorem node9011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9011 :=
  Budget.node_covers _ _ 9011 row9011 node9007 node9029
    node9007_covers node9029_covers (by decide) (by decide) row9011_covers
theorem node9011_eq_data : node9011 = Data.node9011 := by
  have branches : (node9007, node9029) = (Data.node9007, Data.node9029) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9007_eq_data node9029_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9011 row children.1 children.2) row9011_eq_data branches
set_option maxRecDepth 4096 in
theorem row9043_eq_data : row9043 = Data.row9043 := rfl
def node9043 : Graph := .node 9043 row9043 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9043_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9043 :=
  Budget.node_covers _ _ 9043 row9043 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9043_covers
theorem node9043_eq_data : node9043 = Data.node9043 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9043 row children.1 children.2) row9043_eq_data branches
set_option maxRecDepth 4096 in
theorem row9049_eq_data : row9049 = Data.row9049 := rfl
def node9049 : Graph := .node 9049 row9049 node9043 Graph.empty
set_option maxRecDepth 4096 in
theorem node9049_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9049 :=
  Budget.node_covers _ _ 9049 row9049 node9043 Graph.empty
    node9043_covers empty_covers (by decide) (by decide) row9049_covers
theorem node9049_eq_data : node9049 = Data.node9049 := by
  have branches : (node9043, Graph.empty) = (Data.node9043, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9043_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9049 row children.1 children.2) row9049_eq_data branches
set_option maxRecDepth 4096 in
theorem row9067_eq_data : row9067 = Data.row9067 := rfl
def node9067 : Graph := .node 9067 row9067 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9067_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9067 :=
  Budget.node_covers _ _ 9067 row9067 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9067_covers
theorem node9067_eq_data : node9067 = Data.node9067 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9067 row children.1 children.2) row9067_eq_data branches
set_option maxRecDepth 4096 in
theorem row9091_eq_data : row9091 = Data.row9091 := rfl
def node9091 : Graph := .node 9091 row9091 node9067 Graph.empty
set_option maxRecDepth 4096 in
theorem node9091_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9091 :=
  Budget.node_covers _ _ 9091 row9091 node9067 Graph.empty
    node9067_covers empty_covers (by decide) (by decide) row9091_covers
theorem node9091_eq_data : node9091 = Data.node9091 := by
  have branches : (node9067, Graph.empty) = (Data.node9067, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9067_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9091 row children.1 children.2) row9091_eq_data branches
set_option maxRecDepth 4096 in
theorem row9059_eq_data : row9059 = Data.row9059 := rfl
def node9059 : Graph := .node 9059 row9059 node9049 node9091
set_option maxRecDepth 4096 in
theorem node9059_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9059 :=
  Budget.node_covers _ _ 9059 row9059 node9049 node9091
    node9049_covers node9091_covers (by decide) (by decide) row9059_covers
theorem node9059_eq_data : node9059 = Data.node9059 := by
  have branches : (node9049, node9091) = (Data.node9049, Data.node9091) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9049_eq_data node9091_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9059 row children.1 children.2) row9059_eq_data branches
set_option maxRecDepth 4096 in
theorem row9041_eq_data : row9041 = Data.row9041 := rfl
def node9041 : Graph := .node 9041 row9041 node9011 node9059
set_option maxRecDepth 4096 in
theorem node9041_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9041 :=
  Budget.node_covers _ _ 9041 row9041 node9011 node9059
    node9011_covers node9059_covers (by decide) (by decide) row9041_covers
theorem node9041_eq_data : node9041 = Data.node9041 := by
  have branches : (node9011, node9059) = (Data.node9011, Data.node9059) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9011_eq_data node9059_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9041 row children.1 children.2) row9041_eq_data branches
set_option maxRecDepth 4096 in
theorem row9109_eq_data : row9109 = Data.row9109 := rfl
def node9109 : Graph := .node 9109 row9109 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9109_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9109 :=
  Budget.node_covers _ _ 9109 row9109 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9109_covers
theorem node9109_eq_data : node9109 = Data.node9109 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9109 row children.1 children.2) row9109_eq_data branches
set_option maxRecDepth 4096 in
theorem row9127_eq_data : row9127 = Data.row9127 := rfl
def node9127 : Graph := .node 9127 row9127 node9109 Graph.empty
set_option maxRecDepth 4096 in
theorem node9127_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9127 :=
  Budget.node_covers _ _ 9127 row9127 node9109 Graph.empty
    node9109_covers empty_covers (by decide) (by decide) row9127_covers
theorem node9127_eq_data : node9127 = Data.node9127 := by
  have branches : (node9109, Graph.empty) = (Data.node9109, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9109_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9127 row children.1 children.2) row9127_eq_data branches
set_option maxRecDepth 4096 in
theorem row9137_eq_data : row9137 = Data.row9137 := rfl
def node9137 : Graph := .node 9137 row9137 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9137_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9137 :=
  Budget.node_covers _ _ 9137 row9137 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9137_covers
theorem node9137_eq_data : node9137 = Data.node9137 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9137 row children.1 children.2) row9137_eq_data branches
set_option maxRecDepth 4096 in
theorem row9151_eq_data : row9151 = Data.row9151 := rfl
def node9151 : Graph := .node 9151 row9151 node9137 Graph.empty
set_option maxRecDepth 4096 in
theorem node9151_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9151 :=
  Budget.node_covers _ _ 9151 row9151 node9137 Graph.empty
    node9137_covers empty_covers (by decide) (by decide) row9151_covers
theorem node9151_eq_data : node9151 = Data.node9151 := by
  have branches : (node9137, Graph.empty) = (Data.node9137, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9137_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9151 row children.1 children.2) row9151_eq_data branches
set_option maxRecDepth 4096 in
theorem row9133_eq_data : row9133 = Data.row9133 := rfl
def node9133 : Graph := .node 9133 row9133 node9127 node9151
set_option maxRecDepth 4096 in
theorem node9133_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9133 :=
  Budget.node_covers _ _ 9133 row9133 node9127 node9151
    node9127_covers node9151_covers (by decide) (by decide) row9133_covers
theorem node9133_eq_data : node9133 = Data.node9133 := by
  have branches : (node9127, node9151) = (Data.node9127, Data.node9151) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9127_eq_data node9151_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9133 row children.1 children.2) row9133_eq_data branches
set_option maxRecDepth 4096 in
theorem row9161_eq_data : row9161 = Data.row9161 := rfl
def node9161 : Graph := .node 9161 row9161 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9161_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9161 :=
  Budget.node_covers _ _ 9161 row9161 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9161_covers
theorem node9161_eq_data : node9161 = Data.node9161 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9161 row children.1 children.2) row9161_eq_data branches
set_option maxRecDepth 4096 in
theorem row9173_eq_data : row9173 = Data.row9173 := rfl
def node9173 : Graph := .node 9173 row9173 node9161 Graph.empty
set_option maxRecDepth 4096 in
theorem node9173_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9173 :=
  Budget.node_covers _ _ 9173 row9173 node9161 Graph.empty
    node9161_covers empty_covers (by decide) (by decide) row9173_covers
theorem node9173_eq_data : node9173 = Data.node9173 := by
  have branches : (node9161, Graph.empty) = (Data.node9161, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9161_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9173 row children.1 children.2) row9173_eq_data branches
set_option maxRecDepth 4096 in
theorem row9187_eq_data : row9187 = Data.row9187 := rfl
def node9187 : Graph := .node 9187 row9187 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9187_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9187 :=
  Budget.node_covers _ _ 9187 row9187 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9187_covers
theorem node9187_eq_data : node9187 = Data.node9187 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9187 row children.1 children.2) row9187_eq_data branches
set_option maxRecDepth 4096 in
theorem row9199_eq_data : row9199 = Data.row9199 := rfl
def node9199 : Graph := .node 9199 row9199 node9187 Graph.empty
set_option maxRecDepth 4096 in
theorem node9199_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9199 :=
  Budget.node_covers _ _ 9199 row9199 node9187 Graph.empty
    node9187_covers empty_covers (by decide) (by decide) row9199_covers
theorem node9199_eq_data : node9199 = Data.node9199 := by
  have branches : (node9187, Graph.empty) = (Data.node9187, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9187_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9199 row children.1 children.2) row9199_eq_data branches
set_option maxRecDepth 4096 in
theorem row9181_eq_data : row9181 = Data.row9181 := rfl
def node9181 : Graph := .node 9181 row9181 node9173 node9199
set_option maxRecDepth 4096 in
theorem node9181_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9181 :=
  Budget.node_covers _ _ 9181 row9181 node9173 node9199
    node9173_covers node9199_covers (by decide) (by decide) row9181_covers
theorem node9181_eq_data : node9181 = Data.node9181 := by
  have branches : (node9173, node9199) = (Data.node9173, Data.node9199) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9173_eq_data node9199_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9181 row children.1 children.2) row9181_eq_data branches
set_option maxRecDepth 4096 in
theorem row9157_eq_data : row9157 = Data.row9157 := rfl
def node9157 : Graph := .node 9157 row9157 node9133 node9181
set_option maxRecDepth 4096 in
theorem node9157_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9157 :=
  Budget.node_covers _ _ 9157 row9157 node9133 node9181
    node9133_covers node9181_covers (by decide) (by decide) row9157_covers
theorem node9157_eq_data : node9157 = Data.node9157 := by
  have branches : (node9133, node9181) = (Data.node9133, Data.node9181) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9133_eq_data node9181_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9157 row children.1 children.2) row9157_eq_data branches
set_option maxRecDepth 4096 in
theorem row9103_eq_data : row9103 = Data.row9103 := rfl
def node9103 : Graph := .node 9103 row9103 node9041 node9157
set_option maxRecDepth 4096 in
theorem node9103_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9103 :=
  Budget.node_covers _ _ 9103 row9103 node9041 node9157
    node9041_covers node9157_covers (by decide) (by decide) row9103_covers
theorem node9103_eq_data : node9103 = Data.node9103 := by
  have branches : (node9041, node9157) = (Data.node9041, Data.node9157) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9041_eq_data node9157_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9103 row children.1 children.2) row9103_eq_data branches
set_option maxRecDepth 4096 in
theorem row9209_eq_data : row9209 = Data.row9209 := rfl
def node9209 : Graph := .node 9209 row9209 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9209_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9209 :=
  Budget.node_covers _ _ 9209 row9209 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9209_covers
theorem node9209_eq_data : node9209 = Data.node9209 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9209 row children.1 children.2) row9209_eq_data branches
set_option maxRecDepth 4096 in
theorem row9221_eq_data : row9221 = Data.row9221 := rfl
def node9221 : Graph := .node 9221 row9221 node9209 Graph.empty
set_option maxRecDepth 4096 in
theorem node9221_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9221 :=
  Budget.node_covers _ _ 9221 row9221 node9209 Graph.empty
    node9209_covers empty_covers (by decide) (by decide) row9221_covers
theorem node9221_eq_data : node9221 = Data.node9221 := by
  have branches : (node9209, Graph.empty) = (Data.node9209, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9209_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9221 row children.1 children.2) row9221_eq_data branches
set_option maxRecDepth 4096 in
theorem row9239_eq_data : row9239 = Data.row9239 := rfl
def node9239 : Graph := .node 9239 row9239 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9239_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9239 :=
  Budget.node_covers _ _ 9239 row9239 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9239_covers
theorem node9239_eq_data : node9239 = Data.node9239 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9239 row children.1 children.2) row9239_eq_data branches
set_option maxRecDepth 4096 in
theorem row9241_eq_data : row9241 = Data.row9241 := rfl
def node9241 : Graph := .node 9241 row9241 node9239 Graph.empty
set_option maxRecDepth 4096 in
theorem node9241_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9241 :=
  Budget.node_covers _ _ 9241 row9241 node9239 Graph.empty
    node9239_covers empty_covers (by decide) (by decide) row9241_covers
theorem node9241_eq_data : node9241 = Data.node9241 := by
  have branches : (node9239, Graph.empty) = (Data.node9239, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9239_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9241 row children.1 children.2) row9241_eq_data branches
set_option maxRecDepth 4096 in
theorem row9227_eq_data : row9227 = Data.row9227 := rfl
def node9227 : Graph := .node 9227 row9227 node9221 node9241
set_option maxRecDepth 4096 in
theorem node9227_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9227 :=
  Budget.node_covers _ _ 9227 row9227 node9221 node9241
    node9221_covers node9241_covers (by decide) (by decide) row9227_covers
theorem node9227_eq_data : node9227 = Data.node9227 := by
  have branches : (node9221, node9241) = (Data.node9221, Data.node9241) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9221_eq_data node9241_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9227 row children.1 children.2) row9227_eq_data branches
set_option maxRecDepth 4096 in
theorem row9277_eq_data : row9277 = Data.row9277 := rfl
def node9277 : Graph := .node 9277 row9277 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9277_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9277 :=
  Budget.node_covers _ _ 9277 row9277 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9277_covers
theorem node9277_eq_data : node9277 = Data.node9277 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9277 row children.1 children.2) row9277_eq_data branches
set_option maxRecDepth 4096 in
theorem row9281_eq_data : row9281 = Data.row9281 := rfl
def node9281 : Graph := .node 9281 row9281 node9277 Graph.empty
set_option maxRecDepth 4096 in
theorem node9281_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9281 :=
  Budget.node_covers _ _ 9281 row9281 node9277 Graph.empty
    node9277_covers empty_covers (by decide) (by decide) row9281_covers
theorem node9281_eq_data : node9281 = Data.node9281 := by
  have branches : (node9277, Graph.empty) = (Data.node9277, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9277_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9281 row children.1 children.2) row9281_eq_data branches
set_option maxRecDepth 4096 in
theorem row9293_eq_data : row9293 = Data.row9293 := rfl
def node9293 : Graph := .node 9293 row9293 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9293_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9293 :=
  Budget.node_covers _ _ 9293 row9293 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9293_covers
theorem node9293_eq_data : node9293 = Data.node9293 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9293 row children.1 children.2) row9293_eq_data branches
set_option maxRecDepth 4096 in
theorem row9311_eq_data : row9311 = Data.row9311 := rfl
def node9311 : Graph := .node 9311 row9311 node9293 Graph.empty
set_option maxRecDepth 4096 in
theorem node9311_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9311 :=
  Budget.node_covers _ _ 9311 row9311 node9293 Graph.empty
    node9293_covers empty_covers (by decide) (by decide) row9311_covers
theorem node9311_eq_data : node9311 = Data.node9311 := by
  have branches : (node9293, Graph.empty) = (Data.node9293, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9293_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9311 row children.1 children.2) row9311_eq_data branches
set_option maxRecDepth 4096 in
theorem row9283_eq_data : row9283 = Data.row9283 := rfl
def node9283 : Graph := .node 9283 row9283 node9281 node9311
set_option maxRecDepth 4096 in
theorem node9283_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9283 :=
  Budget.node_covers _ _ 9283 row9283 node9281 node9311
    node9281_covers node9311_covers (by decide) (by decide) row9283_covers
theorem node9283_eq_data : node9283 = Data.node9283 := by
  have branches : (node9281, node9311) = (Data.node9281, Data.node9311) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9281_eq_data node9311_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9283 row children.1 children.2) row9283_eq_data branches
set_option maxRecDepth 4096 in
theorem row9257_eq_data : row9257 = Data.row9257 := rfl
def node9257 : Graph := .node 9257 row9257 node9227 node9283
set_option maxRecDepth 4096 in
theorem node9257_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9257 :=
  Budget.node_covers _ _ 9257 row9257 node9227 node9283
    node9227_covers node9283_covers (by decide) (by decide) row9257_covers
theorem node9257_eq_data : node9257 = Data.node9257 := by
  have branches : (node9227, node9283) = (Data.node9227, Data.node9283) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9227_eq_data node9283_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9257 row children.1 children.2) row9257_eq_data branches
set_option maxRecDepth 4096 in
theorem row9323_eq_data : row9323 = Data.row9323 := rfl
def node9323 : Graph := .node 9323 row9323 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9323_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9323 :=
  Budget.node_covers _ _ 9323 row9323 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9323_covers
theorem node9323_eq_data : node9323 = Data.node9323 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9323 row children.1 children.2) row9323_eq_data branches
set_option maxRecDepth 4096 in
theorem row9337_eq_data : row9337 = Data.row9337 := rfl
def node9337 : Graph := .node 9337 row9337 node9323 Graph.empty
set_option maxRecDepth 4096 in
theorem node9337_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9337 :=
  Budget.node_covers _ _ 9337 row9337 node9323 Graph.empty
    node9323_covers empty_covers (by decide) (by decide) row9337_covers
theorem node9337_eq_data : node9337 = Data.node9337 := by
  have branches : (node9323, Graph.empty) = (Data.node9323, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9323_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9337 row children.1 children.2) row9337_eq_data branches
set_option maxRecDepth 4096 in
theorem row9343_eq_data : row9343 = Data.row9343 := rfl
def node9343 : Graph := .node 9343 row9343 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9343_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9343 :=
  Budget.node_covers _ _ 9343 row9343 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9343_covers
theorem node9343_eq_data : node9343 = Data.node9343 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9343 row children.1 children.2) row9343_eq_data branches
set_option maxRecDepth 4096 in
theorem row9349_eq_data : row9349 = Data.row9349 := rfl
def node9349 : Graph := .node 9349 row9349 node9343 Graph.empty
set_option maxRecDepth 4096 in
theorem node9349_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9349 :=
  Budget.node_covers _ _ 9349 row9349 node9343 Graph.empty
    node9343_covers empty_covers (by decide) (by decide) row9349_covers
theorem node9349_eq_data : node9349 = Data.node9349 := by
  have branches : (node9343, Graph.empty) = (Data.node9343, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9343_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9349 row children.1 children.2) row9349_eq_data branches
set_option maxRecDepth 4096 in
theorem row9341_eq_data : row9341 = Data.row9341 := rfl
def node9341 : Graph := .node 9341 row9341 node9337 node9349
set_option maxRecDepth 4096 in
theorem node9341_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9341 :=
  Budget.node_covers _ _ 9341 row9341 node9337 node9349
    node9337_covers node9349_covers (by decide) (by decide) row9341_covers
theorem node9341_eq_data : node9341 = Data.node9341 := by
  have branches : (node9337, node9349) = (Data.node9337, Data.node9349) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9337_eq_data node9349_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9341 row children.1 children.2) row9341_eq_data branches
set_option maxRecDepth 4096 in
theorem row9377_eq_data : row9377 = Data.row9377 := rfl
def node9377 : Graph := .node 9377 row9377 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9377_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9377 :=
  Budget.node_covers _ _ 9377 row9377 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9377_covers
theorem node9377_eq_data : node9377 = Data.node9377 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9377 row children.1 children.2) row9377_eq_data branches
set_option maxRecDepth 4096 in
theorem row9391_eq_data : row9391 = Data.row9391 := rfl
def node9391 : Graph := .node 9391 row9391 node9377 Graph.empty
set_option maxRecDepth 4096 in
theorem node9391_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9391 :=
  Budget.node_covers _ _ 9391 row9391 node9377 Graph.empty
    node9377_covers empty_covers (by decide) (by decide) row9391_covers
theorem node9391_eq_data : node9391 = Data.node9391 := by
  have branches : (node9377, Graph.empty) = (Data.node9377, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9377_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9391 row children.1 children.2) row9391_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
