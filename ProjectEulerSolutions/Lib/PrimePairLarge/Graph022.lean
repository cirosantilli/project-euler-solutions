import ProjectEulerSolutions.Lib.PrimePairLarge.Graph021
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row11813_eq_data : row11813 = Data.row11813 := rfl
def node11813 : Graph := .node 11813 row11813 node11807 Graph.empty
set_option maxRecDepth 4096 in
theorem node11813_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11813 :=
  Budget.node_covers _ _ 11813 row11813 node11807 Graph.empty
    node11807_covers empty_covers (by decide) (by decide) row11813_covers
theorem node11813_eq_data : node11813 = Data.node11813 := by
  have branches : (node11807, Graph.empty) = (Data.node11807, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11807_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11813 row children.1 children.2) row11813_eq_data branches
set_option maxRecDepth 4096 in
theorem row11827_eq_data : row11827 = Data.row11827 := rfl
def node11827 : Graph := .node 11827 row11827 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11827_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11827 :=
  Budget.node_covers _ _ 11827 row11827 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11827_covers
theorem node11827_eq_data : node11827 = Data.node11827 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11827 row children.1 children.2) row11827_eq_data branches
set_option maxRecDepth 4096 in
theorem row11831_eq_data : row11831 = Data.row11831 := rfl
def node11831 : Graph := .node 11831 row11831 node11827 Graph.empty
set_option maxRecDepth 4096 in
theorem node11831_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11831 :=
  Budget.node_covers _ _ 11831 row11831 node11827 Graph.empty
    node11827_covers empty_covers (by decide) (by decide) row11831_covers
theorem node11831_eq_data : node11831 = Data.node11831 := by
  have branches : (node11827, Graph.empty) = (Data.node11827, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11827_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11831 row children.1 children.2) row11831_eq_data branches
set_option maxRecDepth 4096 in
theorem row11821_eq_data : row11821 = Data.row11821 := rfl
def node11821 : Graph := .node 11821 row11821 node11813 node11831
set_option maxRecDepth 4096 in
theorem node11821_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11821 :=
  Budget.node_covers _ _ 11821 row11821 node11813 node11831
    node11813_covers node11831_covers (by decide) (by decide) row11821_covers
theorem node11821_eq_data : node11821 = Data.node11821 := by
  have branches : (node11813, node11831) = (Data.node11813, Data.node11831) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11813_eq_data node11831_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11821 row children.1 children.2) row11821_eq_data branches
set_option maxRecDepth 4096 in
theorem row11801_eq_data : row11801 = Data.row11801 := rfl
def node11801 : Graph := .node 11801 row11801 node11779 node11821
set_option maxRecDepth 4096 in
theorem node11801_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11801 :=
  Budget.node_covers _ _ 11801 row11801 node11779 node11821
    node11779_covers node11821_covers (by decide) (by decide) row11801_covers
theorem node11801_eq_data : node11801 = Data.node11801 := by
  have branches : (node11779, node11821) = (Data.node11779, Data.node11821) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11779_eq_data node11821_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11801 row children.1 children.2) row11801_eq_data branches
set_option maxRecDepth 4096 in
theorem row11839_eq_data : row11839 = Data.row11839 := rfl
def node11839 : Graph := .node 11839 row11839 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11839_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11839 :=
  Budget.node_covers _ _ 11839 row11839 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11839_covers
theorem node11839_eq_data : node11839 = Data.node11839 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11839 row children.1 children.2) row11839_eq_data branches
set_option maxRecDepth 4096 in
theorem row11863_eq_data : row11863 = Data.row11863 := rfl
def node11863 : Graph := .node 11863 row11863 node11839 Graph.empty
set_option maxRecDepth 4096 in
theorem node11863_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11863 :=
  Budget.node_covers _ _ 11863 row11863 node11839 Graph.empty
    node11839_covers empty_covers (by decide) (by decide) row11863_covers
theorem node11863_eq_data : node11863 = Data.node11863 := by
  have branches : (node11839, Graph.empty) = (Data.node11839, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11839_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11863 row children.1 children.2) row11863_eq_data branches
set_option maxRecDepth 4096 in
theorem row11887_eq_data : row11887 = Data.row11887 := rfl
def node11887 : Graph := .node 11887 row11887 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11887_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11887 :=
  Budget.node_covers _ _ 11887 row11887 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11887_covers
theorem node11887_eq_data : node11887 = Data.node11887 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11887 row children.1 children.2) row11887_eq_data branches
set_option maxRecDepth 4096 in
theorem row11897_eq_data : row11897 = Data.row11897 := rfl
def node11897 : Graph := .node 11897 row11897 node11887 Graph.empty
set_option maxRecDepth 4096 in
theorem node11897_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11897 :=
  Budget.node_covers _ _ 11897 row11897 node11887 Graph.empty
    node11887_covers empty_covers (by decide) (by decide) row11897_covers
theorem node11897_eq_data : node11897 = Data.node11897 := by
  have branches : (node11887, Graph.empty) = (Data.node11887, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11887_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11897 row children.1 children.2) row11897_eq_data branches
set_option maxRecDepth 4096 in
theorem row11867_eq_data : row11867 = Data.row11867 := rfl
def node11867 : Graph := .node 11867 row11867 node11863 node11897
set_option maxRecDepth 4096 in
theorem node11867_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11867 :=
  Budget.node_covers _ _ 11867 row11867 node11863 node11897
    node11863_covers node11897_covers (by decide) (by decide) row11867_covers
theorem node11867_eq_data : node11867 = Data.node11867 := by
  have branches : (node11863, node11897) = (Data.node11863, Data.node11897) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11863_eq_data node11897_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11867 row children.1 children.2) row11867_eq_data branches
set_option maxRecDepth 4096 in
theorem row11909_eq_data : row11909 = Data.row11909 := rfl
def node11909 : Graph := .node 11909 row11909 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11909_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11909 :=
  Budget.node_covers _ _ 11909 row11909 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11909_covers
theorem node11909_eq_data : node11909 = Data.node11909 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11909 row children.1 children.2) row11909_eq_data branches
set_option maxRecDepth 4096 in
theorem row11923_eq_data : row11923 = Data.row11923 := rfl
def node11923 : Graph := .node 11923 row11923 node11909 Graph.empty
set_option maxRecDepth 4096 in
theorem node11923_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11923 :=
  Budget.node_covers _ _ 11923 row11923 node11909 Graph.empty
    node11909_covers empty_covers (by decide) (by decide) row11923_covers
theorem node11923_eq_data : node11923 = Data.node11923 := by
  have branches : (node11909, Graph.empty) = (Data.node11909, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11909_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11923 row children.1 children.2) row11923_eq_data branches
set_option maxRecDepth 4096 in
theorem row11933_eq_data : row11933 = Data.row11933 := rfl
def node11933 : Graph := .node 11933 row11933 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11933_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11933 :=
  Budget.node_covers _ _ 11933 row11933 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11933_covers
theorem node11933_eq_data : node11933 = Data.node11933 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11933 row children.1 children.2) row11933_eq_data branches
set_option maxRecDepth 4096 in
theorem row11939_eq_data : row11939 = Data.row11939 := rfl
def node11939 : Graph := .node 11939 row11939 node11933 Graph.empty
set_option maxRecDepth 4096 in
theorem node11939_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11939 :=
  Budget.node_covers _ _ 11939 row11939 node11933 Graph.empty
    node11933_covers empty_covers (by decide) (by decide) row11939_covers
theorem node11939_eq_data : node11939 = Data.node11939 := by
  have branches : (node11933, Graph.empty) = (Data.node11933, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11933_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11939 row children.1 children.2) row11939_eq_data branches
set_option maxRecDepth 4096 in
theorem row11927_eq_data : row11927 = Data.row11927 := rfl
def node11927 : Graph := .node 11927 row11927 node11923 node11939
set_option maxRecDepth 4096 in
theorem node11927_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11927 :=
  Budget.node_covers _ _ 11927 row11927 node11923 node11939
    node11923_covers node11939_covers (by decide) (by decide) row11927_covers
theorem node11927_eq_data : node11927 = Data.node11927 := by
  have branches : (node11923, node11939) = (Data.node11923, Data.node11939) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11923_eq_data node11939_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11927 row children.1 children.2) row11927_eq_data branches
set_option maxRecDepth 4096 in
theorem row11903_eq_data : row11903 = Data.row11903 := rfl
def node11903 : Graph := .node 11903 row11903 node11867 node11927
set_option maxRecDepth 4096 in
theorem node11903_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11903 :=
  Budget.node_covers _ _ 11903 row11903 node11867 node11927
    node11867_covers node11927_covers (by decide) (by decide) row11903_covers
theorem node11903_eq_data : node11903 = Data.node11903 := by
  have branches : (node11867, node11927) = (Data.node11867, Data.node11927) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11867_eq_data node11927_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11903 row children.1 children.2) row11903_eq_data branches
set_option maxRecDepth 4096 in
theorem row11833_eq_data : row11833 = Data.row11833 := rfl
def node11833 : Graph := .node 11833 row11833 node11801 node11903
set_option maxRecDepth 4096 in
theorem node11833_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11833 :=
  Budget.node_covers _ _ 11833 row11833 node11801 node11903
    node11801_covers node11903_covers (by decide) (by decide) row11833_covers
theorem node11833_eq_data : node11833 = Data.node11833 := by
  have branches : (node11801, node11903) = (Data.node11801, Data.node11903) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11801_eq_data node11903_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11833 row children.1 children.2) row11833_eq_data branches
set_option maxRecDepth 4096 in
theorem row11953_eq_data : row11953 = Data.row11953 := rfl
def node11953 : Graph := .node 11953 row11953 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11953_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11953 :=
  Budget.node_covers _ _ 11953 row11953 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11953_covers
theorem node11953_eq_data : node11953 = Data.node11953 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11953 row children.1 children.2) row11953_eq_data branches
set_option maxRecDepth 4096 in
theorem row11959_eq_data : row11959 = Data.row11959 := rfl
def node11959 : Graph := .node 11959 row11959 node11953 Graph.empty
set_option maxRecDepth 4096 in
theorem node11959_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11959 :=
  Budget.node_covers _ _ 11959 row11959 node11953 Graph.empty
    node11953_covers empty_covers (by decide) (by decide) row11959_covers
theorem node11959_eq_data : node11959 = Data.node11959 := by
  have branches : (node11953, Graph.empty) = (Data.node11953, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11953_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11959 row children.1 children.2) row11959_eq_data branches
set_option maxRecDepth 4096 in
theorem row11971_eq_data : row11971 = Data.row11971 := rfl
def node11971 : Graph := .node 11971 row11971 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11971_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11971 :=
  Budget.node_covers _ _ 11971 row11971 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11971_covers
theorem node11971_eq_data : node11971 = Data.node11971 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11971 row children.1 children.2) row11971_eq_data branches
set_option maxRecDepth 4096 in
theorem row11981_eq_data : row11981 = Data.row11981 := rfl
def node11981 : Graph := .node 11981 row11981 node11971 Graph.empty
set_option maxRecDepth 4096 in
theorem node11981_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11981 :=
  Budget.node_covers _ _ 11981 row11981 node11971 Graph.empty
    node11971_covers empty_covers (by decide) (by decide) row11981_covers
theorem node11981_eq_data : node11981 = Data.node11981 := by
  have branches : (node11971, Graph.empty) = (Data.node11971, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11971_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11981 row children.1 children.2) row11981_eq_data branches
set_option maxRecDepth 4096 in
theorem row11969_eq_data : row11969 = Data.row11969 := rfl
def node11969 : Graph := .node 11969 row11969 node11959 node11981
set_option maxRecDepth 4096 in
theorem node11969_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11969 :=
  Budget.node_covers _ _ 11969 row11969 node11959 node11981
    node11959_covers node11981_covers (by decide) (by decide) row11969_covers
theorem node11969_eq_data : node11969 = Data.node11969 := by
  have branches : (node11959, node11981) = (Data.node11959, Data.node11981) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11959_eq_data node11981_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11969 row children.1 children.2) row11969_eq_data branches
set_option maxRecDepth 4096 in
theorem row12007_eq_data : row12007 = Data.row12007 := rfl
def node12007 : Graph := .node 12007 row12007 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12007_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12007 :=
  Budget.node_covers _ _ 12007 row12007 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12007_covers
theorem node12007_eq_data : node12007 = Data.node12007 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12007 row children.1 children.2) row12007_eq_data branches
set_option maxRecDepth 4096 in
theorem row12011_eq_data : row12011 = Data.row12011 := rfl
def node12011 : Graph := .node 12011 row12011 node12007 Graph.empty
set_option maxRecDepth 4096 in
theorem node12011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12011 :=
  Budget.node_covers _ _ 12011 row12011 node12007 Graph.empty
    node12007_covers empty_covers (by decide) (by decide) row12011_covers
theorem node12011_eq_data : node12011 = Data.node12011 := by
  have branches : (node12007, Graph.empty) = (Data.node12007, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12007_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12011 row children.1 children.2) row12011_eq_data branches
set_option maxRecDepth 4096 in
theorem row12041_eq_data : row12041 = Data.row12041 := rfl
def node12041 : Graph := .node 12041 row12041 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12041_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12041 :=
  Budget.node_covers _ _ 12041 row12041 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12041_covers
theorem node12041_eq_data : node12041 = Data.node12041 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12041 row children.1 children.2) row12041_eq_data branches
set_option maxRecDepth 4096 in
theorem row12043_eq_data : row12043 = Data.row12043 := rfl
def node12043 : Graph := .node 12043 row12043 node12041 Graph.empty
set_option maxRecDepth 4096 in
theorem node12043_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12043 :=
  Budget.node_covers _ _ 12043 row12043 node12041 Graph.empty
    node12041_covers empty_covers (by decide) (by decide) row12043_covers
theorem node12043_eq_data : node12043 = Data.node12043 := by
  have branches : (node12041, Graph.empty) = (Data.node12041, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12041_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12043 row children.1 children.2) row12043_eq_data branches
set_option maxRecDepth 4096 in
theorem row12037_eq_data : row12037 = Data.row12037 := rfl
def node12037 : Graph := .node 12037 row12037 node12011 node12043
set_option maxRecDepth 4096 in
theorem node12037_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12037 :=
  Budget.node_covers _ _ 12037 row12037 node12011 node12043
    node12011_covers node12043_covers (by decide) (by decide) row12037_covers
theorem node12037_eq_data : node12037 = Data.node12037 := by
  have branches : (node12011, node12043) = (Data.node12011, Data.node12043) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12011_eq_data node12043_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12037 row children.1 children.2) row12037_eq_data branches
set_option maxRecDepth 4096 in
theorem row11987_eq_data : row11987 = Data.row11987 := rfl
def node11987 : Graph := .node 11987 row11987 node11969 node12037
set_option maxRecDepth 4096 in
theorem node11987_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11987 :=
  Budget.node_covers _ _ 11987 row11987 node11969 node12037
    node11969_covers node12037_covers (by decide) (by decide) row11987_covers
theorem node11987_eq_data : node11987 = Data.node11987 := by
  have branches : (node11969, node12037) = (Data.node11969, Data.node12037) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11969_eq_data node12037_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11987 row children.1 children.2) row11987_eq_data branches
set_option maxRecDepth 4096 in
theorem row12071_eq_data : row12071 = Data.row12071 := rfl
def node12071 : Graph := .node 12071 row12071 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12071_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12071 :=
  Budget.node_covers _ _ 12071 row12071 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12071_covers
theorem node12071_eq_data : node12071 = Data.node12071 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12071 row children.1 children.2) row12071_eq_data branches
set_option maxRecDepth 4096 in
theorem row12073_eq_data : row12073 = Data.row12073 := rfl
def node12073 : Graph := .node 12073 row12073 node12071 Graph.empty
set_option maxRecDepth 4096 in
theorem node12073_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12073 :=
  Budget.node_covers _ _ 12073 row12073 node12071 Graph.empty
    node12071_covers empty_covers (by decide) (by decide) row12073_covers
theorem node12073_eq_data : node12073 = Data.node12073 := by
  have branches : (node12071, Graph.empty) = (Data.node12071, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12071_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12073 row children.1 children.2) row12073_eq_data branches
set_option maxRecDepth 4096 in
theorem row12101_eq_data : row12101 = Data.row12101 := rfl
def node12101 : Graph := .node 12101 row12101 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12101_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12101 :=
  Budget.node_covers _ _ 12101 row12101 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12101_covers
theorem node12101_eq_data : node12101 = Data.node12101 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12101 row children.1 children.2) row12101_eq_data branches
set_option maxRecDepth 4096 in
theorem row12107_eq_data : row12107 = Data.row12107 := rfl
def node12107 : Graph := .node 12107 row12107 node12101 Graph.empty
set_option maxRecDepth 4096 in
theorem node12107_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12107 :=
  Budget.node_covers _ _ 12107 row12107 node12101 Graph.empty
    node12101_covers empty_covers (by decide) (by decide) row12107_covers
theorem node12107_eq_data : node12107 = Data.node12107 := by
  have branches : (node12101, Graph.empty) = (Data.node12101, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12101_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12107 row children.1 children.2) row12107_eq_data branches
set_option maxRecDepth 4096 in
theorem row12097_eq_data : row12097 = Data.row12097 := rfl
def node12097 : Graph := .node 12097 row12097 node12073 node12107
set_option maxRecDepth 4096 in
theorem node12097_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12097 :=
  Budget.node_covers _ _ 12097 row12097 node12073 node12107
    node12073_covers node12107_covers (by decide) (by decide) row12097_covers
theorem node12097_eq_data : node12097 = Data.node12097 := by
  have branches : (node12073, node12107) = (Data.node12073, Data.node12107) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12073_eq_data node12107_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12097 row children.1 children.2) row12097_eq_data branches
set_option maxRecDepth 4096 in
theorem row12113_eq_data : row12113 = Data.row12113 := rfl
def node12113 : Graph := .node 12113 row12113 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12113_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12113 :=
  Budget.node_covers _ _ 12113 row12113 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12113_covers
theorem node12113_eq_data : node12113 = Data.node12113 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12113 row children.1 children.2) row12113_eq_data branches
set_option maxRecDepth 4096 in
theorem row12119_eq_data : row12119 = Data.row12119 := rfl
def node12119 : Graph := .node 12119 row12119 node12113 Graph.empty
set_option maxRecDepth 4096 in
theorem node12119_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12119 :=
  Budget.node_covers _ _ 12119 row12119 node12113 Graph.empty
    node12113_covers empty_covers (by decide) (by decide) row12119_covers
theorem node12119_eq_data : node12119 = Data.node12119 := by
  have branches : (node12113, Graph.empty) = (Data.node12113, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12113_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12119 row children.1 children.2) row12119_eq_data branches
set_option maxRecDepth 4096 in
theorem row12149_eq_data : row12149 = Data.row12149 := rfl
def node12149 : Graph := .node 12149 row12149 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12149_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12149 :=
  Budget.node_covers _ _ 12149 row12149 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12149_covers
theorem node12149_eq_data : node12149 = Data.node12149 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12149 row children.1 children.2) row12149_eq_data branches
set_option maxRecDepth 4096 in
theorem row12157_eq_data : row12157 = Data.row12157 := rfl
def node12157 : Graph := .node 12157 row12157 node12149 Graph.empty
set_option maxRecDepth 4096 in
theorem node12157_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12157 :=
  Budget.node_covers _ _ 12157 row12157 node12149 Graph.empty
    node12149_covers empty_covers (by decide) (by decide) row12157_covers
theorem node12157_eq_data : node12157 = Data.node12157 := by
  have branches : (node12149, Graph.empty) = (Data.node12149, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12149_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12157 row children.1 children.2) row12157_eq_data branches
set_option maxRecDepth 4096 in
theorem row12143_eq_data : row12143 = Data.row12143 := rfl
def node12143 : Graph := .node 12143 row12143 node12119 node12157
set_option maxRecDepth 4096 in
theorem node12143_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12143 :=
  Budget.node_covers _ _ 12143 row12143 node12119 node12157
    node12119_covers node12157_covers (by decide) (by decide) row12143_covers
theorem node12143_eq_data : node12143 = Data.node12143 := by
  have branches : (node12119, node12157) = (Data.node12119, Data.node12157) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12119_eq_data node12157_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12143 row children.1 children.2) row12143_eq_data branches
set_option maxRecDepth 4096 in
theorem row12109_eq_data : row12109 = Data.row12109 := rfl
def node12109 : Graph := .node 12109 row12109 node12097 node12143
set_option maxRecDepth 4096 in
theorem node12109_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12109 :=
  Budget.node_covers _ _ 12109 row12109 node12097 node12143
    node12097_covers node12143_covers (by decide) (by decide) row12109_covers
theorem node12109_eq_data : node12109 = Data.node12109 := by
  have branches : (node12097, node12143) = (Data.node12097, Data.node12143) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12097_eq_data node12143_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12109 row children.1 children.2) row12109_eq_data branches
set_option maxRecDepth 4096 in
theorem row12049_eq_data : row12049 = Data.row12049 := rfl
def node12049 : Graph := .node 12049 row12049 node11987 node12109
set_option maxRecDepth 4096 in
theorem node12049_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12049 :=
  Budget.node_covers _ _ 12049 row12049 node11987 node12109
    node11987_covers node12109_covers (by decide) (by decide) row12049_covers
theorem node12049_eq_data : node12049 = Data.node12049 := by
  have branches : (node11987, node12109) = (Data.node11987, Data.node12109) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11987_eq_data node12109_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12049 row children.1 children.2) row12049_eq_data branches
set_option maxRecDepth 4096 in
theorem row11941_eq_data : row11941 = Data.row11941 := rfl
def node11941 : Graph := .node 11941 row11941 node11833 node12049
set_option maxRecDepth 4096 in
theorem node11941_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11941 :=
  Budget.node_covers _ _ 11941 row11941 node11833 node12049
    node11833_covers node12049_covers (by decide) (by decide) row11941_covers
theorem node11941_eq_data : node11941 = Data.node11941 := by
  have branches : (node11833, node12049) = (Data.node11833, Data.node12049) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11833_eq_data node12049_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11941 row children.1 children.2) row11941_eq_data branches
set_option maxRecDepth 4096 in
theorem row11731_eq_data : row11731 = Data.row11731 := rfl
def node11731 : Graph := .node 11731 row11731 node11483 node11941
set_option maxRecDepth 4096 in
theorem node11731_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11731 :=
  Budget.node_covers _ _ 11731 row11731 node11483 node11941
    node11483_covers node11941_covers (by decide) (by decide) row11731_covers
theorem node11731_eq_data : node11731 = Data.node11731 := by
  have branches : (node11483, node11941) = (Data.node11483, Data.node11941) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11483_eq_data node11941_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11731 row children.1 children.2) row11731_eq_data branches
set_option maxRecDepth 4096 in
theorem row12163_eq_data : row12163 = Data.row12163 := rfl
def node12163 : Graph := .node 12163 row12163 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12163_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12163 :=
  Budget.node_covers _ _ 12163 row12163 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12163_covers
theorem node12163_eq_data : node12163 = Data.node12163 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12163 row children.1 children.2) row12163_eq_data branches
set_option maxRecDepth 4096 in
theorem row12197_eq_data : row12197 = Data.row12197 := rfl
def node12197 : Graph := .node 12197 row12197 node12163 Graph.empty
set_option maxRecDepth 4096 in
theorem node12197_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12197 :=
  Budget.node_covers _ _ 12197 row12197 node12163 Graph.empty
    node12163_covers empty_covers (by decide) (by decide) row12197_covers
theorem node12197_eq_data : node12197 = Data.node12197 := by
  have branches : (node12163, Graph.empty) = (Data.node12163, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12163_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12197 row children.1 children.2) row12197_eq_data branches
set_option maxRecDepth 4096 in
theorem row12211_eq_data : row12211 = Data.row12211 := rfl
def node12211 : Graph := .node 12211 row12211 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12211_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12211 :=
  Budget.node_covers _ _ 12211 row12211 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12211_covers
theorem node12211_eq_data : node12211 = Data.node12211 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12211 row children.1 children.2) row12211_eq_data branches
set_option maxRecDepth 4096 in
theorem row12227_eq_data : row12227 = Data.row12227 := rfl
def node12227 : Graph := .node 12227 row12227 node12211 Graph.empty
set_option maxRecDepth 4096 in
theorem node12227_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12227 :=
  Budget.node_covers _ _ 12227 row12227 node12211 Graph.empty
    node12211_covers empty_covers (by decide) (by decide) row12227_covers
theorem node12227_eq_data : node12227 = Data.node12227 := by
  have branches : (node12211, Graph.empty) = (Data.node12211, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12211_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12227 row children.1 children.2) row12227_eq_data branches
set_option maxRecDepth 4096 in
theorem row12203_eq_data : row12203 = Data.row12203 := rfl
def node12203 : Graph := .node 12203 row12203 node12197 node12227
set_option maxRecDepth 4096 in
theorem node12203_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12203 :=
  Budget.node_covers _ _ 12203 row12203 node12197 node12227
    node12197_covers node12227_covers (by decide) (by decide) row12203_covers
theorem node12203_eq_data : node12203 = Data.node12203 := by
  have branches : (node12197, node12227) = (Data.node12197, Data.node12227) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12197_eq_data node12227_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12203 row children.1 children.2) row12203_eq_data branches
set_option maxRecDepth 4096 in
theorem row12241_eq_data : row12241 = Data.row12241 := rfl
def node12241 : Graph := .node 12241 row12241 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12241_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12241 :=
  Budget.node_covers _ _ 12241 row12241 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12241_covers
theorem node12241_eq_data : node12241 = Data.node12241 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12241 row children.1 children.2) row12241_eq_data branches
set_option maxRecDepth 4096 in
theorem row12251_eq_data : row12251 = Data.row12251 := rfl
def node12251 : Graph := .node 12251 row12251 node12241 Graph.empty
set_option maxRecDepth 4096 in
theorem node12251_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12251 :=
  Budget.node_covers _ _ 12251 row12251 node12241 Graph.empty
    node12241_covers empty_covers (by decide) (by decide) row12251_covers
theorem node12251_eq_data : node12251 = Data.node12251 := by
  have branches : (node12241, Graph.empty) = (Data.node12241, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12241_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12251 row children.1 children.2) row12251_eq_data branches
set_option maxRecDepth 4096 in
theorem row12263_eq_data : row12263 = Data.row12263 := rfl
def node12263 : Graph := .node 12263 row12263 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12263_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12263 :=
  Budget.node_covers _ _ 12263 row12263 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12263_covers
theorem node12263_eq_data : node12263 = Data.node12263 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12263 row children.1 children.2) row12263_eq_data branches
set_option maxRecDepth 4096 in
theorem row12269_eq_data : row12269 = Data.row12269 := rfl
def node12269 : Graph := .node 12269 row12269 node12263 Graph.empty
set_option maxRecDepth 4096 in
theorem node12269_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12269 :=
  Budget.node_covers _ _ 12269 row12269 node12263 Graph.empty
    node12263_covers empty_covers (by decide) (by decide) row12269_covers
theorem node12269_eq_data : node12269 = Data.node12269 := by
  have branches : (node12263, Graph.empty) = (Data.node12263, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12263_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12269 row children.1 children.2) row12269_eq_data branches
set_option maxRecDepth 4096 in
theorem row12253_eq_data : row12253 = Data.row12253 := rfl
def node12253 : Graph := .node 12253 row12253 node12251 node12269
set_option maxRecDepth 4096 in
theorem node12253_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12253 :=
  Budget.node_covers _ _ 12253 row12253 node12251 node12269
    node12251_covers node12269_covers (by decide) (by decide) row12253_covers
theorem node12253_eq_data : node12253 = Data.node12253 := by
  have branches : (node12251, node12269) = (Data.node12251, Data.node12269) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12251_eq_data node12269_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12253 row children.1 children.2) row12253_eq_data branches
set_option maxRecDepth 4096 in
theorem row12239_eq_data : row12239 = Data.row12239 := rfl
def node12239 : Graph := .node 12239 row12239 node12203 node12253
set_option maxRecDepth 4096 in
theorem node12239_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12239 :=
  Budget.node_covers _ _ 12239 row12239 node12203 node12253
    node12203_covers node12253_covers (by decide) (by decide) row12239_covers
theorem node12239_eq_data : node12239 = Data.node12239 := by
  have branches : (node12203, node12253) = (Data.node12203, Data.node12253) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12203_eq_data node12253_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12239 row children.1 children.2) row12239_eq_data branches
set_option maxRecDepth 4096 in
theorem row12281_eq_data : row12281 = Data.row12281 := rfl
def node12281 : Graph := .node 12281 row12281 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12281_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12281 :=
  Budget.node_covers _ _ 12281 row12281 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12281_covers
theorem node12281_eq_data : node12281 = Data.node12281 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12281 row children.1 children.2) row12281_eq_data branches
set_option maxRecDepth 4096 in
theorem row12289_eq_data : row12289 = Data.row12289 := rfl
def node12289 : Graph := .node 12289 row12289 node12281 Graph.empty
set_option maxRecDepth 4096 in
theorem node12289_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12289 :=
  Budget.node_covers _ _ 12289 row12289 node12281 Graph.empty
    node12281_covers empty_covers (by decide) (by decide) row12289_covers
theorem node12289_eq_data : node12289 = Data.node12289 := by
  have branches : (node12281, Graph.empty) = (Data.node12281, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12281_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12289 row children.1 children.2) row12289_eq_data branches
set_option maxRecDepth 4096 in
theorem row12323_eq_data : row12323 = Data.row12323 := rfl
def node12323 : Graph := .node 12323 row12323 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12323_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12323 :=
  Budget.node_covers _ _ 12323 row12323 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12323_covers
theorem node12323_eq_data : node12323 = Data.node12323 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12323 row children.1 children.2) row12323_eq_data branches
set_option maxRecDepth 4096 in
theorem row12329_eq_data : row12329 = Data.row12329 := rfl
def node12329 : Graph := .node 12329 row12329 node12323 Graph.empty
set_option maxRecDepth 4096 in
theorem node12329_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12329 :=
  Budget.node_covers _ _ 12329 row12329 node12323 Graph.empty
    node12323_covers empty_covers (by decide) (by decide) row12329_covers
theorem node12329_eq_data : node12329 = Data.node12329 := by
  have branches : (node12323, Graph.empty) = (Data.node12323, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12323_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12329 row children.1 children.2) row12329_eq_data branches
set_option maxRecDepth 4096 in
theorem row12301_eq_data : row12301 = Data.row12301 := rfl
def node12301 : Graph := .node 12301 row12301 node12289 node12329
set_option maxRecDepth 4096 in
theorem node12301_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12301 :=
  Budget.node_covers _ _ 12301 row12301 node12289 node12329
    node12289_covers node12329_covers (by decide) (by decide) row12301_covers
theorem node12301_eq_data : node12301 = Data.node12301 := by
  have branches : (node12289, node12329) = (Data.node12289, Data.node12329) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12289_eq_data node12329_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12301 row children.1 children.2) row12301_eq_data branches
set_option maxRecDepth 4096 in
theorem row12347_eq_data : row12347 = Data.row12347 := rfl
def node12347 : Graph := .node 12347 row12347 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12347_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12347 :=
  Budget.node_covers _ _ 12347 row12347 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12347_covers
theorem node12347_eq_data : node12347 = Data.node12347 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12347 row children.1 children.2) row12347_eq_data branches
set_option maxRecDepth 4096 in
theorem row12373_eq_data : row12373 = Data.row12373 := rfl
def node12373 : Graph := .node 12373 row12373 node12347 Graph.empty
set_option maxRecDepth 4096 in
theorem node12373_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12373 :=
  Budget.node_covers _ _ 12373 row12373 node12347 Graph.empty
    node12347_covers empty_covers (by decide) (by decide) row12373_covers
theorem node12373_eq_data : node12373 = Data.node12373 := by
  have branches : (node12347, Graph.empty) = (Data.node12347, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12347_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12373 row children.1 children.2) row12373_eq_data branches
set_option maxRecDepth 4096 in
theorem row12379_eq_data : row12379 = Data.row12379 := rfl
def node12379 : Graph := .node 12379 row12379 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12379_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12379 :=
  Budget.node_covers _ _ 12379 row12379 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12379_covers
theorem node12379_eq_data : node12379 = Data.node12379 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12379 row children.1 children.2) row12379_eq_data branches
set_option maxRecDepth 4096 in
theorem row12391_eq_data : row12391 = Data.row12391 := rfl
def node12391 : Graph := .node 12391 row12391 node12379 Graph.empty
set_option maxRecDepth 4096 in
theorem node12391_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12391 :=
  Budget.node_covers _ _ 12391 row12391 node12379 Graph.empty
    node12379_covers empty_covers (by decide) (by decide) row12391_covers
theorem node12391_eq_data : node12391 = Data.node12391 := by
  have branches : (node12379, Graph.empty) = (Data.node12379, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12379_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12391 row children.1 children.2) row12391_eq_data branches
set_option maxRecDepth 4096 in
theorem row12377_eq_data : row12377 = Data.row12377 := rfl
def node12377 : Graph := .node 12377 row12377 node12373 node12391
set_option maxRecDepth 4096 in
theorem node12377_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12377 :=
  Budget.node_covers _ _ 12377 row12377 node12373 node12391
    node12373_covers node12391_covers (by decide) (by decide) row12377_covers
theorem node12377_eq_data : node12377 = Data.node12377 := by
  have branches : (node12373, node12391) = (Data.node12373, Data.node12391) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12373_eq_data node12391_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12377 row children.1 children.2) row12377_eq_data branches
set_option maxRecDepth 4096 in
theorem row12343_eq_data : row12343 = Data.row12343 := rfl
def node12343 : Graph := .node 12343 row12343 node12301 node12377
set_option maxRecDepth 4096 in
theorem node12343_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12343 :=
  Budget.node_covers _ _ 12343 row12343 node12301 node12377
    node12301_covers node12377_covers (by decide) (by decide) row12343_covers
theorem node12343_eq_data : node12343 = Data.node12343 := by
  have branches : (node12301, node12377) = (Data.node12301, Data.node12377) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12301_eq_data node12377_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12343 row children.1 children.2) row12343_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
