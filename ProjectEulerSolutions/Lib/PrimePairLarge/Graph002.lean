import ProjectEulerSolutions.Lib.PrimePairLarge.Graph001
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row761_eq_data : row761 = Data.row761 := rfl
def node761 : Graph := .node 761 row761 node757 Graph.empty
set_option maxRecDepth 4096 in
theorem node761_covers : Budget.Covers PrimePairVertices.largeTree 26034 node761 :=
  Budget.node_covers _ _ 761 row761 node757 Graph.empty
    node757_covers empty_covers (by decide) (by decide) row761_covers
theorem node761_eq_data : node761 = Data.node761 := by
  have branches : (node757, Graph.empty) = (Data.node757, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node757_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 761 row children.1 children.2) row761_eq_data branches
set_option maxRecDepth 4096 in
theorem row751_eq_data : row751 = Data.row751 := rfl
def node751 : Graph := .node 751 row751 node743 node761
set_option maxRecDepth 4096 in
theorem node751_covers : Budget.Covers PrimePairVertices.largeTree 26034 node751 :=
  Budget.node_covers _ _ 751 row751 node743 node761
    node743_covers node761_covers (by decide) (by decide) row751_covers
theorem node751_eq_data : node751 = Data.node751 := by
  have branches : (node743, node761) = (Data.node743, Data.node761) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node743_eq_data node761_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 751 row children.1 children.2) row751_eq_data branches
set_option maxRecDepth 4096 in
theorem row733_eq_data : row733 = Data.row733 := rfl
def node733 : Graph := .node 733 row733 node709 node751
set_option maxRecDepth 4096 in
theorem node733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node733 :=
  Budget.node_covers _ _ 733 row733 node709 node751
    node709_covers node751_covers (by decide) (by decide) row733_covers
theorem node733_eq_data : node733 = Data.node733 := by
  have branches : (node709, node751) = (Data.node709, Data.node751) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node709_eq_data node751_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 733 row children.1 children.2) row733_eq_data branches
set_option maxRecDepth 4096 in
theorem row773_eq_data : row773 = Data.row773 := rfl
def node773 : Graph := .node 773 row773 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node773_covers : Budget.Covers PrimePairVertices.largeTree 26034 node773 :=
  Budget.node_covers _ _ 773 row773 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row773_covers
theorem node773_eq_data : node773 = Data.node773 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 773 row children.1 children.2) row773_eq_data branches
set_option maxRecDepth 4096 in
theorem row787_eq_data : row787 = Data.row787 := rfl
def node787 : Graph := .node 787 row787 node773 Graph.empty
set_option maxRecDepth 4096 in
theorem node787_covers : Budget.Covers PrimePairVertices.largeTree 26034 node787 :=
  Budget.node_covers _ _ 787 row787 node773 Graph.empty
    node773_covers empty_covers (by decide) (by decide) row787_covers
theorem node787_eq_data : node787 = Data.node787 := by
  have branches : (node773, Graph.empty) = (Data.node773, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node773_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 787 row children.1 children.2) row787_eq_data branches
set_option maxRecDepth 4096 in
theorem row809_eq_data : row809 = Data.row809 := rfl
def node809 : Graph := .node 809 row809 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node809_covers : Budget.Covers PrimePairVertices.largeTree 26034 node809 :=
  Budget.node_covers _ _ 809 row809 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row809_covers
theorem node809_eq_data : node809 = Data.node809 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 809 row children.1 children.2) row809_eq_data branches
set_option maxRecDepth 4096 in
theorem row811_eq_data : row811 = Data.row811 := rfl
def node811 : Graph := .node 811 row811 node809 Graph.empty
set_option maxRecDepth 4096 in
theorem node811_covers : Budget.Covers PrimePairVertices.largeTree 26034 node811 :=
  Budget.node_covers _ _ 811 row811 node809 Graph.empty
    node809_covers empty_covers (by decide) (by decide) row811_covers
theorem node811_eq_data : node811 = Data.node811 := by
  have branches : (node809, Graph.empty) = (Data.node809, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node809_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 811 row children.1 children.2) row811_eq_data branches
set_option maxRecDepth 4096 in
theorem row797_eq_data : row797 = Data.row797 := rfl
def node797 : Graph := .node 797 row797 node787 node811
set_option maxRecDepth 4096 in
theorem node797_covers : Budget.Covers PrimePairVertices.largeTree 26034 node797 :=
  Budget.node_covers _ _ 797 row797 node787 node811
    node787_covers node811_covers (by decide) (by decide) row797_covers
theorem node797_eq_data : node797 = Data.node797 := by
  have branches : (node787, node811) = (Data.node787, Data.node811) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node787_eq_data node811_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 797 row children.1 children.2) row797_eq_data branches
set_option maxRecDepth 4096 in
theorem row823_eq_data : row823 = Data.row823 := rfl
def node823 : Graph := .node 823 row823 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node823_covers : Budget.Covers PrimePairVertices.largeTree 26034 node823 :=
  Budget.node_covers _ _ 823 row823 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row823_covers
theorem node823_eq_data : node823 = Data.node823 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 823 row children.1 children.2) row823_eq_data branches
set_option maxRecDepth 4096 in
theorem row827_eq_data : row827 = Data.row827 := rfl
def node827 : Graph := .node 827 row827 node823 Graph.empty
set_option maxRecDepth 4096 in
theorem node827_covers : Budget.Covers PrimePairVertices.largeTree 26034 node827 :=
  Budget.node_covers _ _ 827 row827 node823 Graph.empty
    node823_covers empty_covers (by decide) (by decide) row827_covers
theorem node827_eq_data : node827 = Data.node827 := by
  have branches : (node823, Graph.empty) = (Data.node823, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node823_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 827 row children.1 children.2) row827_eq_data branches
set_option maxRecDepth 4096 in
theorem row839_eq_data : row839 = Data.row839 := rfl
def node839 : Graph := .node 839 row839 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node839_covers : Budget.Covers PrimePairVertices.largeTree 26034 node839 :=
  Budget.node_covers _ _ 839 row839 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row839_covers
theorem node839_eq_data : node839 = Data.node839 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 839 row children.1 children.2) row839_eq_data branches
set_option maxRecDepth 4096 in
theorem row853_eq_data : row853 = Data.row853 := rfl
def node853 : Graph := .node 853 row853 node839 Graph.empty
set_option maxRecDepth 4096 in
theorem node853_covers : Budget.Covers PrimePairVertices.largeTree 26034 node853 :=
  Budget.node_covers _ _ 853 row853 node839 Graph.empty
    node839_covers empty_covers (by decide) (by decide) row853_covers
theorem node853_eq_data : node853 = Data.node853 := by
  have branches : (node839, Graph.empty) = (Data.node839, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node839_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 853 row children.1 children.2) row853_eq_data branches
set_option maxRecDepth 4096 in
theorem row829_eq_data : row829 = Data.row829 := rfl
def node829 : Graph := .node 829 row829 node827 node853
set_option maxRecDepth 4096 in
theorem node829_covers : Budget.Covers PrimePairVertices.largeTree 26034 node829 :=
  Budget.node_covers _ _ 829 row829 node827 node853
    node827_covers node853_covers (by decide) (by decide) row829_covers
theorem node829_eq_data : node829 = Data.node829 := by
  have branches : (node827, node853) = (Data.node827, Data.node853) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node827_eq_data node853_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 829 row children.1 children.2) row829_eq_data branches
set_option maxRecDepth 4096 in
theorem row821_eq_data : row821 = Data.row821 := rfl
def node821 : Graph := .node 821 row821 node797 node829
set_option maxRecDepth 4096 in
theorem node821_covers : Budget.Covers PrimePairVertices.largeTree 26034 node821 :=
  Budget.node_covers _ _ 821 row821 node797 node829
    node797_covers node829_covers (by decide) (by decide) row821_covers
theorem node821_eq_data : node821 = Data.node821 := by
  have branches : (node797, node829) = (Data.node797, Data.node829) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node797_eq_data node829_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 821 row children.1 children.2) row821_eq_data branches
set_option maxRecDepth 4096 in
theorem row769_eq_data : row769 = Data.row769 := rfl
def node769 : Graph := .node 769 row769 node733 node821
set_option maxRecDepth 4096 in
theorem node769_covers : Budget.Covers PrimePairVertices.largeTree 26034 node769 :=
  Budget.node_covers _ _ 769 row769 node733 node821
    node733_covers node821_covers (by decide) (by decide) row769_covers
theorem node769_eq_data : node769 = Data.node769 := by
  have branches : (node733, node821) = (Data.node733, Data.node821) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node733_eq_data node821_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 769 row children.1 children.2) row769_eq_data branches
set_option maxRecDepth 4096 in
theorem row683_eq_data : row683 = Data.row683 := rfl
def node683 : Graph := .node 683 row683 node613 node769
set_option maxRecDepth 4096 in
theorem node683_covers : Budget.Covers PrimePairVertices.largeTree 26034 node683 :=
  Budget.node_covers _ _ 683 row683 node613 node769
    node613_covers node769_covers (by decide) (by decide) row683_covers
theorem node683_eq_data : node683 = Data.node683 := by
  have branches : (node613, node769) = (Data.node613, Data.node769) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node613_eq_data node769_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 683 row children.1 children.2) row683_eq_data branches
set_option maxRecDepth 4096 in
theorem row859_eq_data : row859 = Data.row859 := rfl
def node859 : Graph := .node 859 row859 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node859_covers : Budget.Covers PrimePairVertices.largeTree 26034 node859 :=
  Budget.node_covers _ _ 859 row859 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row859_covers
theorem node859_eq_data : node859 = Data.node859 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 859 row children.1 children.2) row859_eq_data branches
set_option maxRecDepth 4096 in
theorem row863_eq_data : row863 = Data.row863 := rfl
def node863 : Graph := .node 863 row863 node859 Graph.empty
set_option maxRecDepth 4096 in
theorem node863_covers : Budget.Covers PrimePairVertices.largeTree 26034 node863 :=
  Budget.node_covers _ _ 863 row863 node859 Graph.empty
    node859_covers empty_covers (by decide) (by decide) row863_covers
theorem node863_eq_data : node863 = Data.node863 := by
  have branches : (node859, Graph.empty) = (Data.node859, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node859_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 863 row children.1 children.2) row863_eq_data branches
set_option maxRecDepth 4096 in
theorem row881_eq_data : row881 = Data.row881 := rfl
def node881 : Graph := .node 881 row881 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node881_covers : Budget.Covers PrimePairVertices.largeTree 26034 node881 :=
  Budget.node_covers _ _ 881 row881 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row881_covers
theorem node881_eq_data : node881 = Data.node881 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 881 row children.1 children.2) row881_eq_data branches
set_option maxRecDepth 4096 in
theorem row883_eq_data : row883 = Data.row883 := rfl
def node883 : Graph := .node 883 row883 node881 Graph.empty
set_option maxRecDepth 4096 in
theorem node883_covers : Budget.Covers PrimePairVertices.largeTree 26034 node883 :=
  Budget.node_covers _ _ 883 row883 node881 Graph.empty
    node881_covers empty_covers (by decide) (by decide) row883_covers
theorem node883_eq_data : node883 = Data.node883 := by
  have branches : (node881, Graph.empty) = (Data.node881, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node881_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 883 row children.1 children.2) row883_eq_data branches
set_option maxRecDepth 4096 in
theorem row877_eq_data : row877 = Data.row877 := rfl
def node877 : Graph := .node 877 row877 node863 node883
set_option maxRecDepth 4096 in
theorem node877_covers : Budget.Covers PrimePairVertices.largeTree 26034 node877 :=
  Budget.node_covers _ _ 877 row877 node863 node883
    node863_covers node883_covers (by decide) (by decide) row877_covers
theorem node877_eq_data : node877 = Data.node877 := by
  have branches : (node863, node883) = (Data.node863, Data.node883) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node863_eq_data node883_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 877 row children.1 children.2) row877_eq_data branches
set_option maxRecDepth 4096 in
theorem row907_eq_data : row907 = Data.row907 := rfl
def node907 : Graph := .node 907 row907 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node907 :=
  Budget.node_covers _ _ 907 row907 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row907_covers
theorem node907_eq_data : node907 = Data.node907 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 907 row children.1 children.2) row907_eq_data branches
set_option maxRecDepth 4096 in
theorem row911_eq_data : row911 = Data.row911 := rfl
def node911 : Graph := .node 911 row911 node907 Graph.empty
set_option maxRecDepth 4096 in
theorem node911_covers : Budget.Covers PrimePairVertices.largeTree 26034 node911 :=
  Budget.node_covers _ _ 911 row911 node907 Graph.empty
    node907_covers empty_covers (by decide) (by decide) row911_covers
theorem node911_eq_data : node911 = Data.node911 := by
  have branches : (node907, Graph.empty) = (Data.node907, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node907_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 911 row children.1 children.2) row911_eq_data branches
set_option maxRecDepth 4096 in
theorem row929_eq_data : row929 = Data.row929 := rfl
def node929 : Graph := .node 929 row929 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node929_covers : Budget.Covers PrimePairVertices.largeTree 26034 node929 :=
  Budget.node_covers _ _ 929 row929 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row929_covers
theorem node929_eq_data : node929 = Data.node929 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 929 row children.1 children.2) row929_eq_data branches
set_option maxRecDepth 4096 in
theorem row937_eq_data : row937 = Data.row937 := rfl
def node937 : Graph := .node 937 row937 node929 Graph.empty
set_option maxRecDepth 4096 in
theorem node937_covers : Budget.Covers PrimePairVertices.largeTree 26034 node937 :=
  Budget.node_covers _ _ 937 row937 node929 Graph.empty
    node929_covers empty_covers (by decide) (by decide) row937_covers
theorem node937_eq_data : node937 = Data.node937 := by
  have branches : (node929, Graph.empty) = (Data.node929, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node929_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 937 row children.1 children.2) row937_eq_data branches
set_option maxRecDepth 4096 in
theorem row919_eq_data : row919 = Data.row919 := rfl
def node919 : Graph := .node 919 row919 node911 node937
set_option maxRecDepth 4096 in
theorem node919_covers : Budget.Covers PrimePairVertices.largeTree 26034 node919 :=
  Budget.node_covers _ _ 919 row919 node911 node937
    node911_covers node937_covers (by decide) (by decide) row919_covers
theorem node919_eq_data : node919 = Data.node919 := by
  have branches : (node911, node937) = (Data.node911, Data.node937) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node911_eq_data node937_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 919 row children.1 children.2) row919_eq_data branches
set_option maxRecDepth 4096 in
theorem row887_eq_data : row887 = Data.row887 := rfl
def node887 : Graph := .node 887 row887 node877 node919
set_option maxRecDepth 4096 in
theorem node887_covers : Budget.Covers PrimePairVertices.largeTree 26034 node887 :=
  Budget.node_covers _ _ 887 row887 node877 node919
    node877_covers node919_covers (by decide) (by decide) row887_covers
theorem node887_eq_data : node887 = Data.node887 := by
  have branches : (node877, node919) = (Data.node877, Data.node919) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node877_eq_data node919_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 887 row children.1 children.2) row887_eq_data branches
set_option maxRecDepth 4096 in
theorem row947_eq_data : row947 = Data.row947 := rfl
def node947 : Graph := .node 947 row947 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node947_covers : Budget.Covers PrimePairVertices.largeTree 26034 node947 :=
  Budget.node_covers _ _ 947 row947 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row947_covers
theorem node947_eq_data : node947 = Data.node947 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 947 row children.1 children.2) row947_eq_data branches
set_option maxRecDepth 4096 in
theorem row953_eq_data : row953 = Data.row953 := rfl
def node953 : Graph := .node 953 row953 node947 Graph.empty
set_option maxRecDepth 4096 in
theorem node953_covers : Budget.Covers PrimePairVertices.largeTree 26034 node953 :=
  Budget.node_covers _ _ 953 row953 node947 Graph.empty
    node947_covers empty_covers (by decide) (by decide) row953_covers
theorem node953_eq_data : node953 = Data.node953 := by
  have branches : (node947, Graph.empty) = (Data.node947, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node947_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 953 row children.1 children.2) row953_eq_data branches
set_option maxRecDepth 4096 in
theorem row971_eq_data : row971 = Data.row971 := rfl
def node971 : Graph := .node 971 row971 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node971_covers : Budget.Covers PrimePairVertices.largeTree 26034 node971 :=
  Budget.node_covers _ _ 971 row971 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row971_covers
theorem node971_eq_data : node971 = Data.node971 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 971 row children.1 children.2) row971_eq_data branches
set_option maxRecDepth 4096 in
theorem row977_eq_data : row977 = Data.row977 := rfl
def node977 : Graph := .node 977 row977 node971 Graph.empty
set_option maxRecDepth 4096 in
theorem node977_covers : Budget.Covers PrimePairVertices.largeTree 26034 node977 :=
  Budget.node_covers _ _ 977 row977 node971 Graph.empty
    node971_covers empty_covers (by decide) (by decide) row977_covers
theorem node977_eq_data : node977 = Data.node977 := by
  have branches : (node971, Graph.empty) = (Data.node971, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node971_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 977 row children.1 children.2) row977_eq_data branches
set_option maxRecDepth 4096 in
theorem row967_eq_data : row967 = Data.row967 := rfl
def node967 : Graph := .node 967 row967 node953 node977
set_option maxRecDepth 4096 in
theorem node967_covers : Budget.Covers PrimePairVertices.largeTree 26034 node967 :=
  Budget.node_covers _ _ 967 row967 node953 node977
    node953_covers node977_covers (by decide) (by decide) row967_covers
theorem node967_eq_data : node967 = Data.node967 := by
  have branches : (node953, node977) = (Data.node953, Data.node977) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node953_eq_data node977_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 967 row children.1 children.2) row967_eq_data branches
set_option maxRecDepth 4096 in
theorem row991_eq_data : row991 = Data.row991 := rfl
def node991 : Graph := .node 991 row991 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node991_covers : Budget.Covers PrimePairVertices.largeTree 26034 node991 :=
  Budget.node_covers _ _ 991 row991 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row991_covers
theorem node991_eq_data : node991 = Data.node991 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 991 row children.1 children.2) row991_eq_data branches
set_option maxRecDepth 4096 in
theorem row997_eq_data : row997 = Data.row997 := rfl
def node997 : Graph := .node 997 row997 node991 Graph.empty
set_option maxRecDepth 4096 in
theorem node997_covers : Budget.Covers PrimePairVertices.largeTree 26034 node997 :=
  Budget.node_covers _ _ 997 row997 node991 Graph.empty
    node991_covers empty_covers (by decide) (by decide) row997_covers
theorem node997_eq_data : node997 = Data.node997 := by
  have branches : (node991, Graph.empty) = (Data.node991, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node991_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 997 row children.1 children.2) row997_eq_data branches
set_option maxRecDepth 4096 in
theorem row1013_eq_data : row1013 = Data.row1013 := rfl
def node1013 : Graph := .node 1013 row1013 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1013_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1013 :=
  Budget.node_covers _ _ 1013 row1013 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1013_covers
theorem node1013_eq_data : node1013 = Data.node1013 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1013 row children.1 children.2) row1013_eq_data branches
set_option maxRecDepth 4096 in
theorem row1019_eq_data : row1019 = Data.row1019 := rfl
def node1019 : Graph := .node 1019 row1019 node1013 Graph.empty
set_option maxRecDepth 4096 in
theorem node1019_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1019 :=
  Budget.node_covers _ _ 1019 row1019 node1013 Graph.empty
    node1013_covers empty_covers (by decide) (by decide) row1019_covers
theorem node1019_eq_data : node1019 = Data.node1019 := by
  have branches : (node1013, Graph.empty) = (Data.node1013, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1013_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1019 row children.1 children.2) row1019_eq_data branches
set_option maxRecDepth 4096 in
theorem row1009_eq_data : row1009 = Data.row1009 := rfl
def node1009 : Graph := .node 1009 row1009 node997 node1019
set_option maxRecDepth 4096 in
theorem node1009_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1009 :=
  Budget.node_covers _ _ 1009 row1009 node997 node1019
    node997_covers node1019_covers (by decide) (by decide) row1009_covers
theorem node1009_eq_data : node1009 = Data.node1009 := by
  have branches : (node997, node1019) = (Data.node997, Data.node1019) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node997_eq_data node1019_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1009 row children.1 children.2) row1009_eq_data branches
set_option maxRecDepth 4096 in
theorem row983_eq_data : row983 = Data.row983 := rfl
def node983 : Graph := .node 983 row983 node967 node1009
set_option maxRecDepth 4096 in
theorem node983_covers : Budget.Covers PrimePairVertices.largeTree 26034 node983 :=
  Budget.node_covers _ _ 983 row983 node967 node1009
    node967_covers node1009_covers (by decide) (by decide) row983_covers
theorem node983_eq_data : node983 = Data.node983 := by
  have branches : (node967, node1009) = (Data.node967, Data.node1009) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node967_eq_data node1009_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 983 row children.1 children.2) row983_eq_data branches
set_option maxRecDepth 4096 in
theorem row941_eq_data : row941 = Data.row941 := rfl
def node941 : Graph := .node 941 row941 node887 node983
set_option maxRecDepth 4096 in
theorem node941_covers : Budget.Covers PrimePairVertices.largeTree 26034 node941 :=
  Budget.node_covers _ _ 941 row941 node887 node983
    node887_covers node983_covers (by decide) (by decide) row941_covers
theorem node941_eq_data : node941 = Data.node941 := by
  have branches : (node887, node983) = (Data.node887, Data.node983) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node887_eq_data node983_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 941 row children.1 children.2) row941_eq_data branches
set_option maxRecDepth 4096 in
theorem row1031_eq_data : row1031 = Data.row1031 := rfl
def node1031 : Graph := .node 1031 row1031 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1031_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1031 :=
  Budget.node_covers _ _ 1031 row1031 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1031_covers
theorem node1031_eq_data : node1031 = Data.node1031 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1031 row children.1 children.2) row1031_eq_data branches
set_option maxRecDepth 4096 in
theorem row1033_eq_data : row1033 = Data.row1033 := rfl
def node1033 : Graph := .node 1033 row1033 node1031 Graph.empty
set_option maxRecDepth 4096 in
theorem node1033_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1033 :=
  Budget.node_covers _ _ 1033 row1033 node1031 Graph.empty
    node1031_covers empty_covers (by decide) (by decide) row1033_covers
theorem node1033_eq_data : node1033 = Data.node1033 := by
  have branches : (node1031, Graph.empty) = (Data.node1031, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1031_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1033 row children.1 children.2) row1033_eq_data branches
set_option maxRecDepth 4096 in
theorem row1049_eq_data : row1049 = Data.row1049 := rfl
def node1049 : Graph := .node 1049 row1049 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1049_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1049 :=
  Budget.node_covers _ _ 1049 row1049 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1049_covers
theorem node1049_eq_data : node1049 = Data.node1049 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1049 row children.1 children.2) row1049_eq_data branches
set_option maxRecDepth 4096 in
theorem row1051_eq_data : row1051 = Data.row1051 := rfl
def node1051 : Graph := .node 1051 row1051 node1049 Graph.empty
set_option maxRecDepth 4096 in
theorem node1051_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1051 :=
  Budget.node_covers _ _ 1051 row1051 node1049 Graph.empty
    node1049_covers empty_covers (by decide) (by decide) row1051_covers
theorem node1051_eq_data : node1051 = Data.node1051 := by
  have branches : (node1049, Graph.empty) = (Data.node1049, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1049_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1051 row children.1 children.2) row1051_eq_data branches
set_option maxRecDepth 4096 in
theorem row1039_eq_data : row1039 = Data.row1039 := rfl
def node1039 : Graph := .node 1039 row1039 node1033 node1051
set_option maxRecDepth 4096 in
theorem node1039_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1039 :=
  Budget.node_covers _ _ 1039 row1039 node1033 node1051
    node1033_covers node1051_covers (by decide) (by decide) row1039_covers
theorem node1039_eq_data : node1039 = Data.node1039 := by
  have branches : (node1033, node1051) = (Data.node1033, Data.node1051) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1033_eq_data node1051_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1039 row children.1 children.2) row1039_eq_data branches
set_option maxRecDepth 4096 in
theorem row1063_eq_data : row1063 = Data.row1063 := rfl
def node1063 : Graph := .node 1063 row1063 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1063_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1063 :=
  Budget.node_covers _ _ 1063 row1063 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1063_covers
theorem node1063_eq_data : node1063 = Data.node1063 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1063 row children.1 children.2) row1063_eq_data branches
set_option maxRecDepth 4096 in
theorem row1069_eq_data : row1069 = Data.row1069 := rfl
def node1069 : Graph := .node 1069 row1069 node1063 Graph.empty
set_option maxRecDepth 4096 in
theorem node1069_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1069 :=
  Budget.node_covers _ _ 1069 row1069 node1063 Graph.empty
    node1063_covers empty_covers (by decide) (by decide) row1069_covers
theorem node1069_eq_data : node1069 = Data.node1069 := by
  have branches : (node1063, Graph.empty) = (Data.node1063, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1063_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1069 row children.1 children.2) row1069_eq_data branches
set_option maxRecDepth 4096 in
theorem row1091_eq_data : row1091 = Data.row1091 := rfl
def node1091 : Graph := .node 1091 row1091 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1091_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1091 :=
  Budget.node_covers _ _ 1091 row1091 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1091_covers
theorem node1091_eq_data : node1091 = Data.node1091 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1091 row children.1 children.2) row1091_eq_data branches
set_option maxRecDepth 4096 in
theorem row1093_eq_data : row1093 = Data.row1093 := rfl
def node1093 : Graph := .node 1093 row1093 node1091 Graph.empty
set_option maxRecDepth 4096 in
theorem node1093_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1093 :=
  Budget.node_covers _ _ 1093 row1093 node1091 Graph.empty
    node1091_covers empty_covers (by decide) (by decide) row1093_covers
theorem node1093_eq_data : node1093 = Data.node1093 := by
  have branches : (node1091, Graph.empty) = (Data.node1091, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1091_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1093 row children.1 children.2) row1093_eq_data branches
set_option maxRecDepth 4096 in
theorem row1087_eq_data : row1087 = Data.row1087 := rfl
def node1087 : Graph := .node 1087 row1087 node1069 node1093
set_option maxRecDepth 4096 in
theorem node1087_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1087 :=
  Budget.node_covers _ _ 1087 row1087 node1069 node1093
    node1069_covers node1093_covers (by decide) (by decide) row1087_covers
theorem node1087_eq_data : node1087 = Data.node1087 := by
  have branches : (node1069, node1093) = (Data.node1069, Data.node1093) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1069_eq_data node1093_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1087 row children.1 children.2) row1087_eq_data branches
set_option maxRecDepth 4096 in
theorem row1061_eq_data : row1061 = Data.row1061 := rfl
def node1061 : Graph := .node 1061 row1061 node1039 node1087
set_option maxRecDepth 4096 in
theorem node1061_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1061 :=
  Budget.node_covers _ _ 1061 row1061 node1039 node1087
    node1039_covers node1087_covers (by decide) (by decide) row1061_covers
theorem node1061_eq_data : node1061 = Data.node1061 := by
  have branches : (node1039, node1087) = (Data.node1039, Data.node1087) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1039_eq_data node1087_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1061 row children.1 children.2) row1061_eq_data branches
set_option maxRecDepth 4096 in
theorem row1103_eq_data : row1103 = Data.row1103 := rfl
def node1103 : Graph := .node 1103 row1103 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1103_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1103 :=
  Budget.node_covers _ _ 1103 row1103 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1103_covers
theorem node1103_eq_data : node1103 = Data.node1103 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1103 row children.1 children.2) row1103_eq_data branches
set_option maxRecDepth 4096 in
theorem row1109_eq_data : row1109 = Data.row1109 := rfl
def node1109 : Graph := .node 1109 row1109 node1103 Graph.empty
set_option maxRecDepth 4096 in
theorem node1109_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1109 :=
  Budget.node_covers _ _ 1109 row1109 node1103 Graph.empty
    node1103_covers empty_covers (by decide) (by decide) row1109_covers
theorem node1109_eq_data : node1109 = Data.node1109 := by
  have branches : (node1103, Graph.empty) = (Data.node1103, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1103_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1109 row children.1 children.2) row1109_eq_data branches
set_option maxRecDepth 4096 in
theorem row1123_eq_data : row1123 = Data.row1123 := rfl
def node1123 : Graph := .node 1123 row1123 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1123_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1123 :=
  Budget.node_covers _ _ 1123 row1123 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1123_covers
theorem node1123_eq_data : node1123 = Data.node1123 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1123 row children.1 children.2) row1123_eq_data branches
set_option maxRecDepth 4096 in
theorem row1129_eq_data : row1129 = Data.row1129 := rfl
def node1129 : Graph := .node 1129 row1129 node1123 Graph.empty
set_option maxRecDepth 4096 in
theorem node1129_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1129 :=
  Budget.node_covers _ _ 1129 row1129 node1123 Graph.empty
    node1123_covers empty_covers (by decide) (by decide) row1129_covers
theorem node1129_eq_data : node1129 = Data.node1129 := by
  have branches : (node1123, Graph.empty) = (Data.node1123, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1123_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1129 row children.1 children.2) row1129_eq_data branches
set_option maxRecDepth 4096 in
theorem row1117_eq_data : row1117 = Data.row1117 := rfl
def node1117 : Graph := .node 1117 row1117 node1109 node1129
set_option maxRecDepth 4096 in
theorem node1117_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1117 :=
  Budget.node_covers _ _ 1117 row1117 node1109 node1129
    node1109_covers node1129_covers (by decide) (by decide) row1117_covers
theorem node1117_eq_data : node1117 = Data.node1117 := by
  have branches : (node1109, node1129) = (Data.node1109, Data.node1129) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1109_eq_data node1129_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1117 row children.1 children.2) row1117_eq_data branches
set_option maxRecDepth 4096 in
theorem row1153_eq_data : row1153 = Data.row1153 := rfl
def node1153 : Graph := .node 1153 row1153 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1153_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1153 :=
  Budget.node_covers _ _ 1153 row1153 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1153_covers
theorem node1153_eq_data : node1153 = Data.node1153 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1153 row children.1 children.2) row1153_eq_data branches
set_option maxRecDepth 4096 in
theorem row1163_eq_data : row1163 = Data.row1163 := rfl
def node1163 : Graph := .node 1163 row1163 node1153 Graph.empty
set_option maxRecDepth 4096 in
theorem node1163_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1163 :=
  Budget.node_covers _ _ 1163 row1163 node1153 Graph.empty
    node1153_covers empty_covers (by decide) (by decide) row1163_covers
theorem node1163_eq_data : node1163 = Data.node1163 := by
  have branches : (node1153, Graph.empty) = (Data.node1153, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1153_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1163 row children.1 children.2) row1163_eq_data branches
set_option maxRecDepth 4096 in
theorem row1181_eq_data : row1181 = Data.row1181 := rfl
def node1181 : Graph := .node 1181 row1181 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1181_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1181 :=
  Budget.node_covers _ _ 1181 row1181 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1181_covers
theorem node1181_eq_data : node1181 = Data.node1181 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1181 row children.1 children.2) row1181_eq_data branches
set_option maxRecDepth 4096 in
theorem row1187_eq_data : row1187 = Data.row1187 := rfl
def node1187 : Graph := .node 1187 row1187 node1181 Graph.empty
set_option maxRecDepth 4096 in
theorem node1187_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1187 :=
  Budget.node_covers _ _ 1187 row1187 node1181 Graph.empty
    node1181_covers empty_covers (by decide) (by decide) row1187_covers
theorem node1187_eq_data : node1187 = Data.node1187 := by
  have branches : (node1181, Graph.empty) = (Data.node1181, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1181_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1187 row children.1 children.2) row1187_eq_data branches
set_option maxRecDepth 4096 in
theorem row1171_eq_data : row1171 = Data.row1171 := rfl
def node1171 : Graph := .node 1171 row1171 node1163 node1187
set_option maxRecDepth 4096 in
theorem node1171_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1171 :=
  Budget.node_covers _ _ 1171 row1171 node1163 node1187
    node1163_covers node1187_covers (by decide) (by decide) row1171_covers
theorem node1171_eq_data : node1171 = Data.node1171 := by
  have branches : (node1163, node1187) = (Data.node1163, Data.node1187) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1163_eq_data node1187_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1171 row children.1 children.2) row1171_eq_data branches
set_option maxRecDepth 4096 in
theorem row1151_eq_data : row1151 = Data.row1151 := rfl
def node1151 : Graph := .node 1151 row1151 node1117 node1171
set_option maxRecDepth 4096 in
theorem node1151_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1151 :=
  Budget.node_covers _ _ 1151 row1151 node1117 node1171
    node1117_covers node1171_covers (by decide) (by decide) row1151_covers
theorem node1151_eq_data : node1151 = Data.node1151 := by
  have branches : (node1117, node1171) = (Data.node1117, Data.node1171) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1117_eq_data node1171_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1151 row children.1 children.2) row1151_eq_data branches
set_option maxRecDepth 4096 in
theorem row1097_eq_data : row1097 = Data.row1097 := rfl
def node1097 : Graph := .node 1097 row1097 node1061 node1151
set_option maxRecDepth 4096 in
theorem node1097_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1097 :=
  Budget.node_covers _ _ 1097 row1097 node1061 node1151
    node1061_covers node1151_covers (by decide) (by decide) row1097_covers
theorem node1097_eq_data : node1097 = Data.node1097 := by
  have branches : (node1061, node1151) = (Data.node1061, Data.node1151) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1061_eq_data node1151_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1097 row children.1 children.2) row1097_eq_data branches
set_option maxRecDepth 4096 in
theorem row1021_eq_data : row1021 = Data.row1021 := rfl
def node1021 : Graph := .node 1021 row1021 node941 node1097
set_option maxRecDepth 4096 in
theorem node1021_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1021 :=
  Budget.node_covers _ _ 1021 row1021 node941 node1097
    node941_covers node1097_covers (by decide) (by decide) row1021_covers
theorem node1021_eq_data : node1021 = Data.node1021 := by
  have branches : (node941, node1097) = (Data.node941, Data.node1097) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node941_eq_data node1097_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1021 row children.1 children.2) row1021_eq_data branches
set_option maxRecDepth 4096 in
theorem row857_eq_data : row857 = Data.row857 := rfl
def node857 : Graph := .node 857 row857 node683 node1021
set_option maxRecDepth 4096 in
theorem node857_covers : Budget.Covers PrimePairVertices.largeTree 26034 node857 :=
  Budget.node_covers _ _ 857 row857 node683 node1021
    node683_covers node1021_covers (by decide) (by decide) row857_covers
theorem node857_eq_data : node857 = Data.node857 := by
  have branches : (node683, node1021) = (Data.node683, Data.node1021) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node683_eq_data node1021_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 857 row children.1 children.2) row857_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
