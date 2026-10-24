import ProjectEulerSolutions.Lib.PrimePairLarge.Graph011
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row5827_eq_data : row5827 = Data.row5827 := rfl
def node5827 : Graph := .node 5827 row5827 node5783 node5861
set_option maxRecDepth 4096 in
theorem node5827_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5827 :=
  Budget.node_covers _ _ 5827 row5827 node5783 node5861
    node5783_covers node5861_covers (by decide) (by decide) row5827_covers
theorem node5827_eq_data : node5827 = Data.node5827 := by
  have branches : (node5783, node5861) = (Data.node5783, Data.node5861) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5783_eq_data node5861_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5827 row children.1 children.2) row5827_eq_data branches
set_option maxRecDepth 4096 in
theorem row5717_eq_data : row5717 = Data.row5717 := rfl
def node5717 : Graph := .node 5717 row5717 node5641 node5827
set_option maxRecDepth 4096 in
theorem node5717_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5717 :=
  Budget.node_covers _ _ 5717 row5717 node5641 node5827
    node5641_covers node5827_covers (by decide) (by decide) row5717_covers
theorem node5717_eq_data : node5717 = Data.node5717 := by
  have branches : (node5641, node5827) = (Data.node5641, Data.node5827) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5641_eq_data node5827_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5717 row children.1 children.2) row5717_eq_data branches
set_option maxRecDepth 4096 in
theorem row5519_eq_data : row5519 = Data.row5519 := rfl
def node5519 : Graph := .node 5519 row5519 node5323 node5717
set_option maxRecDepth 4096 in
theorem node5519_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5519 :=
  Budget.node_covers _ _ 5519 row5519 node5323 node5717
    node5323_covers node5717_covers (by decide) (by decide) row5519_covers
theorem node5519_eq_data : node5519 = Data.node5519 := by
  have branches : (node5323, node5717) = (Data.node5323, Data.node5717) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5323_eq_data node5717_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5519 row children.1 children.2) row5519_eq_data branches
set_option maxRecDepth 4096 in
theorem row5099_eq_data : row5099 = Data.row5099 := rfl
def node5099 : Graph := .node 5099 row5099 node4679 node5519
set_option maxRecDepth 4096 in
theorem node5099_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5099 :=
  Budget.node_covers _ _ 5099 row5099 node4679 node5519
    node4679_covers node5519_covers (by decide) (by decide) row5099_covers
theorem node5099_eq_data : node5099 = Data.node5099 := by
  have branches : (node4679, node5519) = (Data.node4679, Data.node5519) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4679_eq_data node5519_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5099 row children.1 children.2) row5099_eq_data branches
set_option maxRecDepth 4096 in
theorem row4259_eq_data : row4259 = Data.row4259 := rfl
def node4259 : Graph := .node 4259 row4259 node3469 node5099
set_option maxRecDepth 4096 in
theorem node4259_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4259 :=
  Budget.node_covers _ _ 4259 row4259 node3469 node5099
    node3469_covers node5099_covers (by decide) (by decide) row4259_covers
theorem node4259_eq_data : node4259 = Data.node4259 := by
  have branches : (node3469, node5099) = (Data.node3469, Data.node5099) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3469_eq_data node5099_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4259 row children.1 children.2) row4259_eq_data branches
set_option maxRecDepth 4096 in
theorem row2687_eq_data : row2687 = Data.row2687 := rfl
def node2687 : Graph := .node 2687 row2687 node1193 node4259
set_option maxRecDepth 4096 in
theorem node2687_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2687 :=
  Budget.node_covers _ _ 2687 row2687 node1193 node4259
    node1193_covers node4259_covers (by decide) (by decide) row2687_covers
theorem node2687_eq_data : node2687 = Data.node2687 := by
  have branches : (node1193, node4259) = (Data.node1193, Data.node4259) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1193_eq_data node4259_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2687 row children.1 children.2) row2687_eq_data branches
set_option maxRecDepth 4096 in
theorem row5923_eq_data : row5923 = Data.row5923 := rfl
def node5923 : Graph := .node 5923 row5923 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5923_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5923 :=
  Budget.node_covers _ _ 5923 row5923 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5923_covers
theorem node5923_eq_data : node5923 = Data.node5923 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5923 row children.1 children.2) row5923_eq_data branches
set_option maxRecDepth 4096 in
theorem row5939_eq_data : row5939 = Data.row5939 := rfl
def node5939 : Graph := .node 5939 row5939 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5939_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5939 :=
  Budget.node_covers _ _ 5939 row5939 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5939_covers
theorem node5939_eq_data : node5939 = Data.node5939 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5939 row children.1 children.2) row5939_eq_data branches
set_option maxRecDepth 4096 in
theorem row5927_eq_data : row5927 = Data.row5927 := rfl
def node5927 : Graph := .node 5927 row5927 node5923 node5939
set_option maxRecDepth 4096 in
theorem node5927_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5927 :=
  Budget.node_covers _ _ 5927 row5927 node5923 node5939
    node5923_covers node5939_covers (by decide) (by decide) row5927_covers
theorem node5927_eq_data : node5927 = Data.node5927 := by
  have branches : (node5923, node5939) = (Data.node5923, Data.node5939) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5923_eq_data node5939_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5927 row children.1 children.2) row5927_eq_data branches
set_option maxRecDepth 4096 in
theorem row5981_eq_data : row5981 = Data.row5981 := rfl
def node5981 : Graph := .node 5981 row5981 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5981_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5981 :=
  Budget.node_covers _ _ 5981 row5981 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5981_covers
theorem node5981_eq_data : node5981 = Data.node5981 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5981 row children.1 children.2) row5981_eq_data branches
set_option maxRecDepth 4096 in
theorem row5987_eq_data : row5987 = Data.row5987 := rfl
def node5987 : Graph := .node 5987 row5987 node5981 Graph.empty
set_option maxRecDepth 4096 in
theorem node5987_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5987 :=
  Budget.node_covers _ _ 5987 row5987 node5981 Graph.empty
    node5981_covers empty_covers (by decide) (by decide) row5987_covers
theorem node5987_eq_data : node5987 = Data.node5987 := by
  have branches : (node5981, Graph.empty) = (Data.node5981, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5981_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5987 row children.1 children.2) row5987_eq_data branches
set_option maxRecDepth 4096 in
theorem row5953_eq_data : row5953 = Data.row5953 := rfl
def node5953 : Graph := .node 5953 row5953 node5927 node5987
set_option maxRecDepth 4096 in
theorem node5953_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5953 :=
  Budget.node_covers _ _ 5953 row5953 node5927 node5987
    node5927_covers node5987_covers (by decide) (by decide) row5953_covers
theorem node5953_eq_data : node5953 = Data.node5953 := by
  have branches : (node5927, node5987) = (Data.node5927, Data.node5987) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5927_eq_data node5987_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5953 row children.1 children.2) row5953_eq_data branches
set_option maxRecDepth 4096 in
theorem row6011_eq_data : row6011 = Data.row6011 := rfl
def node6011 : Graph := .node 6011 row6011 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6011 :=
  Budget.node_covers _ _ 6011 row6011 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6011_covers
theorem node6011_eq_data : node6011 = Data.node6011 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6011 row children.1 children.2) row6011_eq_data branches
set_option maxRecDepth 4096 in
theorem row6029_eq_data : row6029 = Data.row6029 := rfl
def node6029 : Graph := .node 6029 row6029 node6011 Graph.empty
set_option maxRecDepth 4096 in
theorem node6029_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6029 :=
  Budget.node_covers _ _ 6029 row6029 node6011 Graph.empty
    node6011_covers empty_covers (by decide) (by decide) row6029_covers
theorem node6029_eq_data : node6029 = Data.node6029 := by
  have branches : (node6011, Graph.empty) = (Data.node6011, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6011_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6029 row children.1 children.2) row6029_eq_data branches
set_option maxRecDepth 4096 in
theorem row6043_eq_data : row6043 = Data.row6043 := rfl
def node6043 : Graph := .node 6043 row6043 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6043_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6043 :=
  Budget.node_covers _ _ 6043 row6043 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6043_covers
theorem node6043_eq_data : node6043 = Data.node6043 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6043 row children.1 children.2) row6043_eq_data branches
set_option maxRecDepth 4096 in
theorem row6047_eq_data : row6047 = Data.row6047 := rfl
def node6047 : Graph := .node 6047 row6047 node6043 Graph.empty
set_option maxRecDepth 4096 in
theorem node6047_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6047 :=
  Budget.node_covers _ _ 6047 row6047 node6043 Graph.empty
    node6043_covers empty_covers (by decide) (by decide) row6047_covers
theorem node6047_eq_data : node6047 = Data.node6047 := by
  have branches : (node6043, Graph.empty) = (Data.node6043, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6043_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6047 row children.1 children.2) row6047_eq_data branches
set_option maxRecDepth 4096 in
theorem row6037_eq_data : row6037 = Data.row6037 := rfl
def node6037 : Graph := .node 6037 row6037 node6029 node6047
set_option maxRecDepth 4096 in
theorem node6037_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6037 :=
  Budget.node_covers _ _ 6037 row6037 node6029 node6047
    node6029_covers node6047_covers (by decide) (by decide) row6037_covers
theorem node6037_eq_data : node6037 = Data.node6037 := by
  have branches : (node6029, node6047) = (Data.node6029, Data.node6047) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6029_eq_data node6047_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6037 row children.1 children.2) row6037_eq_data branches
set_option maxRecDepth 4096 in
theorem row6007_eq_data : row6007 = Data.row6007 := rfl
def node6007 : Graph := .node 6007 row6007 node5953 node6037
set_option maxRecDepth 4096 in
theorem node6007_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6007 :=
  Budget.node_covers _ _ 6007 row6007 node5953 node6037
    node5953_covers node6037_covers (by decide) (by decide) row6007_covers
theorem node6007_eq_data : node6007 = Data.node6007 := by
  have branches : (node5953, node6037) = (Data.node5953, Data.node6037) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5953_eq_data node6037_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6007 row children.1 children.2) row6007_eq_data branches
set_option maxRecDepth 4096 in
theorem row6067_eq_data : row6067 = Data.row6067 := rfl
def node6067 : Graph := .node 6067 row6067 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6067_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6067 :=
  Budget.node_covers _ _ 6067 row6067 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6067_covers
theorem node6067_eq_data : node6067 = Data.node6067 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6067 row children.1 children.2) row6067_eq_data branches
set_option maxRecDepth 4096 in
theorem row6073_eq_data : row6073 = Data.row6073 := rfl
def node6073 : Graph := .node 6073 row6073 node6067 Graph.empty
set_option maxRecDepth 4096 in
theorem node6073_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6073 :=
  Budget.node_covers _ _ 6073 row6073 node6067 Graph.empty
    node6067_covers empty_covers (by decide) (by decide) row6073_covers
theorem node6073_eq_data : node6073 = Data.node6073 := by
  have branches : (node6067, Graph.empty) = (Data.node6067, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6067_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6073 row children.1 children.2) row6073_eq_data branches
set_option maxRecDepth 4096 in
theorem row6089_eq_data : row6089 = Data.row6089 := rfl
def node6089 : Graph := .node 6089 row6089 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6089_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6089 :=
  Budget.node_covers _ _ 6089 row6089 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6089_covers
theorem node6089_eq_data : node6089 = Data.node6089 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6089 row children.1 children.2) row6089_eq_data branches
set_option maxRecDepth 4096 in
theorem row6091_eq_data : row6091 = Data.row6091 := rfl
def node6091 : Graph := .node 6091 row6091 node6089 Graph.empty
set_option maxRecDepth 4096 in
theorem node6091_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6091 :=
  Budget.node_covers _ _ 6091 row6091 node6089 Graph.empty
    node6089_covers empty_covers (by decide) (by decide) row6091_covers
theorem node6091_eq_data : node6091 = Data.node6091 := by
  have branches : (node6089, Graph.empty) = (Data.node6089, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6089_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6091 row children.1 children.2) row6091_eq_data branches
set_option maxRecDepth 4096 in
theorem row6079_eq_data : row6079 = Data.row6079 := rfl
def node6079 : Graph := .node 6079 row6079 node6073 node6091
set_option maxRecDepth 4096 in
theorem node6079_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6079 :=
  Budget.node_covers _ _ 6079 row6079 node6073 node6091
    node6073_covers node6091_covers (by decide) (by decide) row6079_covers
theorem node6079_eq_data : node6079 = Data.node6079 := by
  have branches : (node6073, node6091) = (Data.node6073, Data.node6091) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6073_eq_data node6091_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6079 row children.1 children.2) row6079_eq_data branches
set_option maxRecDepth 4096 in
theorem row6113_eq_data : row6113 = Data.row6113 := rfl
def node6113 : Graph := .node 6113 row6113 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6113_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6113 :=
  Budget.node_covers _ _ 6113 row6113 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6113_covers
theorem node6113_eq_data : node6113 = Data.node6113 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6113 row children.1 children.2) row6113_eq_data branches
set_option maxRecDepth 4096 in
theorem row6121_eq_data : row6121 = Data.row6121 := rfl
def node6121 : Graph := .node 6121 row6121 node6113 Graph.empty
set_option maxRecDepth 4096 in
theorem node6121_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6121 :=
  Budget.node_covers _ _ 6121 row6121 node6113 Graph.empty
    node6113_covers empty_covers (by decide) (by decide) row6121_covers
theorem node6121_eq_data : node6121 = Data.node6121 := by
  have branches : (node6113, Graph.empty) = (Data.node6113, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6113_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6121 row children.1 children.2) row6121_eq_data branches
set_option maxRecDepth 4096 in
theorem row6133_eq_data : row6133 = Data.row6133 := rfl
def node6133 : Graph := .node 6133 row6133 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6133_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6133 :=
  Budget.node_covers _ _ 6133 row6133 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6133_covers
theorem node6133_eq_data : node6133 = Data.node6133 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6133 row children.1 children.2) row6133_eq_data branches
set_option maxRecDepth 4096 in
theorem row6143_eq_data : row6143 = Data.row6143 := rfl
def node6143 : Graph := .node 6143 row6143 node6133 Graph.empty
set_option maxRecDepth 4096 in
theorem node6143_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6143 :=
  Budget.node_covers _ _ 6143 row6143 node6133 Graph.empty
    node6133_covers empty_covers (by decide) (by decide) row6143_covers
theorem node6143_eq_data : node6143 = Data.node6143 := by
  have branches : (node6133, Graph.empty) = (Data.node6133, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6133_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6143 row children.1 children.2) row6143_eq_data branches
set_option maxRecDepth 4096 in
theorem row6131_eq_data : row6131 = Data.row6131 := rfl
def node6131 : Graph := .node 6131 row6131 node6121 node6143
set_option maxRecDepth 4096 in
theorem node6131_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6131 :=
  Budget.node_covers _ _ 6131 row6131 node6121 node6143
    node6121_covers node6143_covers (by decide) (by decide) row6131_covers
theorem node6131_eq_data : node6131 = Data.node6131 := by
  have branches : (node6121, node6143) = (Data.node6121, Data.node6143) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6121_eq_data node6143_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6131 row children.1 children.2) row6131_eq_data branches
set_option maxRecDepth 4096 in
theorem row6101_eq_data : row6101 = Data.row6101 := rfl
def node6101 : Graph := .node 6101 row6101 node6079 node6131
set_option maxRecDepth 4096 in
theorem node6101_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6101 :=
  Budget.node_covers _ _ 6101 row6101 node6079 node6131
    node6079_covers node6131_covers (by decide) (by decide) row6101_covers
theorem node6101_eq_data : node6101 = Data.node6101 := by
  have branches : (node6079, node6131) = (Data.node6079, Data.node6131) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6079_eq_data node6131_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6101 row children.1 children.2) row6101_eq_data branches
set_option maxRecDepth 4096 in
theorem row6053_eq_data : row6053 = Data.row6053 := rfl
def node6053 : Graph := .node 6053 row6053 node6007 node6101
set_option maxRecDepth 4096 in
theorem node6053_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6053 :=
  Budget.node_covers _ _ 6053 row6053 node6007 node6101
    node6007_covers node6101_covers (by decide) (by decide) row6053_covers
theorem node6053_eq_data : node6053 = Data.node6053 := by
  have branches : (node6007, node6101) = (Data.node6007, Data.node6101) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6007_eq_data node6101_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6053 row children.1 children.2) row6053_eq_data branches
set_option maxRecDepth 4096 in
theorem row6163_eq_data : row6163 = Data.row6163 := rfl
def node6163 : Graph := .node 6163 row6163 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6163_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6163 :=
  Budget.node_covers _ _ 6163 row6163 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6163_covers
theorem node6163_eq_data : node6163 = Data.node6163 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6163 row children.1 children.2) row6163_eq_data branches
set_option maxRecDepth 4096 in
theorem row6173_eq_data : row6173 = Data.row6173 := rfl
def node6173 : Graph := .node 6173 row6173 node6163 Graph.empty
set_option maxRecDepth 4096 in
theorem node6173_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6173 :=
  Budget.node_covers _ _ 6173 row6173 node6163 Graph.empty
    node6163_covers empty_covers (by decide) (by decide) row6173_covers
theorem node6173_eq_data : node6173 = Data.node6173 := by
  have branches : (node6163, Graph.empty) = (Data.node6163, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6163_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6173 row children.1 children.2) row6173_eq_data branches
set_option maxRecDepth 4096 in
theorem row6199_eq_data : row6199 = Data.row6199 := rfl
def node6199 : Graph := .node 6199 row6199 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6199_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6199 :=
  Budget.node_covers _ _ 6199 row6199 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6199_covers
theorem node6199_eq_data : node6199 = Data.node6199 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6199 row children.1 children.2) row6199_eq_data branches
set_option maxRecDepth 4096 in
theorem row6203_eq_data : row6203 = Data.row6203 := rfl
def node6203 : Graph := .node 6203 row6203 node6199 Graph.empty
set_option maxRecDepth 4096 in
theorem node6203_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6203 :=
  Budget.node_covers _ _ 6203 row6203 node6199 Graph.empty
    node6199_covers empty_covers (by decide) (by decide) row6203_covers
theorem node6203_eq_data : node6203 = Data.node6203 := by
  have branches : (node6199, Graph.empty) = (Data.node6199, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6199_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6203 row children.1 children.2) row6203_eq_data branches
set_option maxRecDepth 4096 in
theorem row6197_eq_data : row6197 = Data.row6197 := rfl
def node6197 : Graph := .node 6197 row6197 node6173 node6203
set_option maxRecDepth 4096 in
theorem node6197_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6197 :=
  Budget.node_covers _ _ 6197 row6197 node6173 node6203
    node6173_covers node6203_covers (by decide) (by decide) row6197_covers
theorem node6197_eq_data : node6197 = Data.node6197 := by
  have branches : (node6173, node6203) = (Data.node6173, Data.node6203) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6173_eq_data node6203_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6197 row children.1 children.2) row6197_eq_data branches
set_option maxRecDepth 4096 in
theorem row6217_eq_data : row6217 = Data.row6217 := rfl
def node6217 : Graph := .node 6217 row6217 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6217_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6217 :=
  Budget.node_covers _ _ 6217 row6217 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6217_covers
theorem node6217_eq_data : node6217 = Data.node6217 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6217 row children.1 children.2) row6217_eq_data branches
set_option maxRecDepth 4096 in
theorem row6221_eq_data : row6221 = Data.row6221 := rfl
def node6221 : Graph := .node 6221 row6221 node6217 Graph.empty
set_option maxRecDepth 4096 in
theorem node6221_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6221 :=
  Budget.node_covers _ _ 6221 row6221 node6217 Graph.empty
    node6217_covers empty_covers (by decide) (by decide) row6221_covers
theorem node6221_eq_data : node6221 = Data.node6221 := by
  have branches : (node6217, Graph.empty) = (Data.node6217, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6217_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6221 row children.1 children.2) row6221_eq_data branches
set_option maxRecDepth 4096 in
theorem row6247_eq_data : row6247 = Data.row6247 := rfl
def node6247 : Graph := .node 6247 row6247 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6247_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6247 :=
  Budget.node_covers _ _ 6247 row6247 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6247_covers
theorem node6247_eq_data : node6247 = Data.node6247 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6247 row children.1 children.2) row6247_eq_data branches
set_option maxRecDepth 4096 in
theorem row6257_eq_data : row6257 = Data.row6257 := rfl
def node6257 : Graph := .node 6257 row6257 node6247 Graph.empty
set_option maxRecDepth 4096 in
theorem node6257_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6257 :=
  Budget.node_covers _ _ 6257 row6257 node6247 Graph.empty
    node6247_covers empty_covers (by decide) (by decide) row6257_covers
theorem node6257_eq_data : node6257 = Data.node6257 := by
  have branches : (node6247, Graph.empty) = (Data.node6247, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6247_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6257 row children.1 children.2) row6257_eq_data branches
set_option maxRecDepth 4096 in
theorem row6229_eq_data : row6229 = Data.row6229 := rfl
def node6229 : Graph := .node 6229 row6229 node6221 node6257
set_option maxRecDepth 4096 in
theorem node6229_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6229 :=
  Budget.node_covers _ _ 6229 row6229 node6221 node6257
    node6221_covers node6257_covers (by decide) (by decide) row6229_covers
theorem node6229_eq_data : node6229 = Data.node6229 := by
  have branches : (node6221, node6257) = (Data.node6221, Data.node6257) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6221_eq_data node6257_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6229 row children.1 children.2) row6229_eq_data branches
set_option maxRecDepth 4096 in
theorem row6211_eq_data : row6211 = Data.row6211 := rfl
def node6211 : Graph := .node 6211 row6211 node6197 node6229
set_option maxRecDepth 4096 in
theorem node6211_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6211 :=
  Budget.node_covers _ _ 6211 row6211 node6197 node6229
    node6197_covers node6229_covers (by decide) (by decide) row6211_covers
theorem node6211_eq_data : node6211 = Data.node6211 := by
  have branches : (node6197, node6229) = (Data.node6197, Data.node6229) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6197_eq_data node6229_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6211 row children.1 children.2) row6211_eq_data branches
set_option maxRecDepth 4096 in
theorem row6269_eq_data : row6269 = Data.row6269 := rfl
def node6269 : Graph := .node 6269 row6269 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6269_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6269 :=
  Budget.node_covers _ _ 6269 row6269 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6269_covers
theorem node6269_eq_data : node6269 = Data.node6269 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6269 row children.1 children.2) row6269_eq_data branches
set_option maxRecDepth 4096 in
theorem row6271_eq_data : row6271 = Data.row6271 := rfl
def node6271 : Graph := .node 6271 row6271 node6269 Graph.empty
set_option maxRecDepth 4096 in
theorem node6271_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6271 :=
  Budget.node_covers _ _ 6271 row6271 node6269 Graph.empty
    node6269_covers empty_covers (by decide) (by decide) row6271_covers
theorem node6271_eq_data : node6271 = Data.node6271 := by
  have branches : (node6269, Graph.empty) = (Data.node6269, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6269_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6271 row children.1 children.2) row6271_eq_data branches
set_option maxRecDepth 4096 in
theorem row6287_eq_data : row6287 = Data.row6287 := rfl
def node6287 : Graph := .node 6287 row6287 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6287_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6287 :=
  Budget.node_covers _ _ 6287 row6287 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6287_covers
theorem node6287_eq_data : node6287 = Data.node6287 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6287 row children.1 children.2) row6287_eq_data branches
set_option maxRecDepth 4096 in
theorem row6299_eq_data : row6299 = Data.row6299 := rfl
def node6299 : Graph := .node 6299 row6299 node6287 Graph.empty
set_option maxRecDepth 4096 in
theorem node6299_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6299 :=
  Budget.node_covers _ _ 6299 row6299 node6287 Graph.empty
    node6287_covers empty_covers (by decide) (by decide) row6299_covers
theorem node6299_eq_data : node6299 = Data.node6299 := by
  have branches : (node6287, Graph.empty) = (Data.node6287, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6287_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6299 row children.1 children.2) row6299_eq_data branches
set_option maxRecDepth 4096 in
theorem row6277_eq_data : row6277 = Data.row6277 := rfl
def node6277 : Graph := .node 6277 row6277 node6271 node6299
set_option maxRecDepth 4096 in
theorem node6277_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6277 :=
  Budget.node_covers _ _ 6277 row6277 node6271 node6299
    node6271_covers node6299_covers (by decide) (by decide) row6277_covers
theorem node6277_eq_data : node6277 = Data.node6277 := by
  have branches : (node6271, node6299) = (Data.node6271, Data.node6299) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6271_eq_data node6299_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6277 row children.1 children.2) row6277_eq_data branches
set_option maxRecDepth 4096 in
theorem row6311_eq_data : row6311 = Data.row6311 := rfl
def node6311 : Graph := .node 6311 row6311 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6311_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6311 :=
  Budget.node_covers _ _ 6311 row6311 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6311_covers
theorem node6311_eq_data : node6311 = Data.node6311 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6311 row children.1 children.2) row6311_eq_data branches
set_option maxRecDepth 4096 in
theorem row6317_eq_data : row6317 = Data.row6317 := rfl
def node6317 : Graph := .node 6317 row6317 node6311 Graph.empty
set_option maxRecDepth 4096 in
theorem node6317_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6317 :=
  Budget.node_covers _ _ 6317 row6317 node6311 Graph.empty
    node6311_covers empty_covers (by decide) (by decide) row6317_covers
theorem node6317_eq_data : node6317 = Data.node6317 := by
  have branches : (node6311, Graph.empty) = (Data.node6311, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6311_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6317 row children.1 children.2) row6317_eq_data branches
set_option maxRecDepth 4096 in
theorem row6329_eq_data : row6329 = Data.row6329 := rfl
def node6329 : Graph := .node 6329 row6329 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6329_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6329 :=
  Budget.node_covers _ _ 6329 row6329 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6329_covers
theorem node6329_eq_data : node6329 = Data.node6329 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6329 row children.1 children.2) row6329_eq_data branches
set_option maxRecDepth 4096 in
theorem row6337_eq_data : row6337 = Data.row6337 := rfl
def node6337 : Graph := .node 6337 row6337 node6329 Graph.empty
set_option maxRecDepth 4096 in
theorem node6337_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6337 :=
  Budget.node_covers _ _ 6337 row6337 node6329 Graph.empty
    node6329_covers empty_covers (by decide) (by decide) row6337_covers
theorem node6337_eq_data : node6337 = Data.node6337 := by
  have branches : (node6329, Graph.empty) = (Data.node6329, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6329_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6337 row children.1 children.2) row6337_eq_data branches
set_option maxRecDepth 4096 in
theorem row6323_eq_data : row6323 = Data.row6323 := rfl
def node6323 : Graph := .node 6323 row6323 node6317 node6337
set_option maxRecDepth 4096 in
theorem node6323_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6323 :=
  Budget.node_covers _ _ 6323 row6323 node6317 node6337
    node6317_covers node6337_covers (by decide) (by decide) row6323_covers
theorem node6323_eq_data : node6323 = Data.node6323 := by
  have branches : (node6317, node6337) = (Data.node6317, Data.node6337) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6317_eq_data node6337_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6323 row children.1 children.2) row6323_eq_data branches
set_option maxRecDepth 4096 in
theorem row6301_eq_data : row6301 = Data.row6301 := rfl
def node6301 : Graph := .node 6301 row6301 node6277 node6323
set_option maxRecDepth 4096 in
theorem node6301_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6301 :=
  Budget.node_covers _ _ 6301 row6301 node6277 node6323
    node6277_covers node6323_covers (by decide) (by decide) row6301_covers
theorem node6301_eq_data : node6301 = Data.node6301 := by
  have branches : (node6277, node6323) = (Data.node6277, Data.node6323) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6277_eq_data node6323_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6301 row children.1 children.2) row6301_eq_data branches
set_option maxRecDepth 4096 in
theorem row6263_eq_data : row6263 = Data.row6263 := rfl
def node6263 : Graph := .node 6263 row6263 node6211 node6301
set_option maxRecDepth 4096 in
theorem node6263_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6263 :=
  Budget.node_covers _ _ 6263 row6263 node6211 node6301
    node6211_covers node6301_covers (by decide) (by decide) row6263_covers
theorem node6263_eq_data : node6263 = Data.node6263 := by
  have branches : (node6211, node6301) = (Data.node6211, Data.node6301) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6211_eq_data node6301_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6263 row children.1 children.2) row6263_eq_data branches
set_option maxRecDepth 4096 in
theorem row6151_eq_data : row6151 = Data.row6151 := rfl
def node6151 : Graph := .node 6151 row6151 node6053 node6263
set_option maxRecDepth 4096 in
theorem node6151_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6151 :=
  Budget.node_covers _ _ 6151 row6151 node6053 node6263
    node6053_covers node6263_covers (by decide) (by decide) row6151_covers
theorem node6151_eq_data : node6151 = Data.node6151 := by
  have branches : (node6053, node6263) = (Data.node6053, Data.node6263) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6053_eq_data node6263_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6151 row children.1 children.2) row6151_eq_data branches
set_option maxRecDepth 4096 in
theorem row6353_eq_data : row6353 = Data.row6353 := rfl
def node6353 : Graph := .node 6353 row6353 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6353_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6353 :=
  Budget.node_covers _ _ 6353 row6353 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6353_covers
theorem node6353_eq_data : node6353 = Data.node6353 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6353 row children.1 children.2) row6353_eq_data branches
set_option maxRecDepth 4096 in
theorem row6359_eq_data : row6359 = Data.row6359 := rfl
def node6359 : Graph := .node 6359 row6359 node6353 Graph.empty
set_option maxRecDepth 4096 in
theorem node6359_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6359 :=
  Budget.node_covers _ _ 6359 row6359 node6353 Graph.empty
    node6353_covers empty_covers (by decide) (by decide) row6359_covers
theorem node6359_eq_data : node6359 = Data.node6359 := by
  have branches : (node6353, Graph.empty) = (Data.node6353, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6353_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6359 row children.1 children.2) row6359_eq_data branches
set_option maxRecDepth 4096 in
theorem row6367_eq_data : row6367 = Data.row6367 := rfl
def node6367 : Graph := .node 6367 row6367 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6367_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6367 :=
  Budget.node_covers _ _ 6367 row6367 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6367_covers
theorem node6367_eq_data : node6367 = Data.node6367 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6367 row children.1 children.2) row6367_eq_data branches
set_option maxRecDepth 4096 in
theorem row6373_eq_data : row6373 = Data.row6373 := rfl
def node6373 : Graph := .node 6373 row6373 node6367 Graph.empty
set_option maxRecDepth 4096 in
theorem node6373_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6373 :=
  Budget.node_covers _ _ 6373 row6373 node6367 Graph.empty
    node6367_covers empty_covers (by decide) (by decide) row6373_covers
theorem node6373_eq_data : node6373 = Data.node6373 := by
  have branches : (node6367, Graph.empty) = (Data.node6367, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6367_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6373 row children.1 children.2) row6373_eq_data branches
set_option maxRecDepth 4096 in
theorem row6361_eq_data : row6361 = Data.row6361 := rfl
def node6361 : Graph := .node 6361 row6361 node6359 node6373
set_option maxRecDepth 4096 in
theorem node6361_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6361 :=
  Budget.node_covers _ _ 6361 row6361 node6359 node6373
    node6359_covers node6373_covers (by decide) (by decide) row6361_covers
theorem node6361_eq_data : node6361 = Data.node6361 := by
  have branches : (node6359, node6373) = (Data.node6359, Data.node6373) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6359_eq_data node6373_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6361 row children.1 children.2) row6361_eq_data branches
set_option maxRecDepth 4096 in
theorem row6389_eq_data : row6389 = Data.row6389 := rfl
def node6389 : Graph := .node 6389 row6389 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6389_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6389 :=
  Budget.node_covers _ _ 6389 row6389 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6389_covers
theorem node6389_eq_data : node6389 = Data.node6389 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6389 row children.1 children.2) row6389_eq_data branches
set_option maxRecDepth 4096 in
theorem row6397_eq_data : row6397 = Data.row6397 := rfl
def node6397 : Graph := .node 6397 row6397 node6389 Graph.empty
set_option maxRecDepth 4096 in
theorem node6397_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6397 :=
  Budget.node_covers _ _ 6397 row6397 node6389 Graph.empty
    node6389_covers empty_covers (by decide) (by decide) row6397_covers
theorem node6397_eq_data : node6397 = Data.node6397 := by
  have branches : (node6389, Graph.empty) = (Data.node6389, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6389_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6397 row children.1 children.2) row6397_eq_data branches
set_option maxRecDepth 4096 in
theorem row6427_eq_data : row6427 = Data.row6427 := rfl
def node6427 : Graph := .node 6427 row6427 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6427_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6427 :=
  Budget.node_covers _ _ 6427 row6427 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6427_covers
theorem node6427_eq_data : node6427 = Data.node6427 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6427 row children.1 children.2) row6427_eq_data branches
set_option maxRecDepth 4096 in
theorem row6449_eq_data : row6449 = Data.row6449 := rfl
def node6449 : Graph := .node 6449 row6449 node6427 Graph.empty
set_option maxRecDepth 4096 in
theorem node6449_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6449 :=
  Budget.node_covers _ _ 6449 row6449 node6427 Graph.empty
    node6427_covers empty_covers (by decide) (by decide) row6449_covers
theorem node6449_eq_data : node6449 = Data.node6449 := by
  have branches : (node6427, Graph.empty) = (Data.node6427, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6427_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6449 row children.1 children.2) row6449_eq_data branches
set_option maxRecDepth 4096 in
theorem row6421_eq_data : row6421 = Data.row6421 := rfl
def node6421 : Graph := .node 6421 row6421 node6397 node6449
set_option maxRecDepth 4096 in
theorem node6421_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6421 :=
  Budget.node_covers _ _ 6421 row6421 node6397 node6449
    node6397_covers node6449_covers (by decide) (by decide) row6421_covers
theorem node6421_eq_data : node6421 = Data.node6421 := by
  have branches : (node6397, node6449) = (Data.node6397, Data.node6449) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6397_eq_data node6449_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6421 row children.1 children.2) row6421_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
