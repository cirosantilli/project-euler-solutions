import ProjectEulerSolutions.Lib.PrimePairLarge.Graph020
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row11173_eq_data : row11173 = Data.row11173 := rfl
def node11173 : Graph := .node 11173 row11173 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11173_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11173 :=
  Budget.node_covers _ _ 11173 row11173 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11173_covers
theorem node11173_eq_data : node11173 = Data.node11173 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11173 row children.1 children.2) row11173_eq_data branches
set_option maxRecDepth 4096 in
theorem row11177_eq_data : row11177 = Data.row11177 := rfl
def node11177 : Graph := .node 11177 row11177 node11173 Graph.empty
set_option maxRecDepth 4096 in
theorem node11177_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11177 :=
  Budget.node_covers _ _ 11177 row11177 node11173 Graph.empty
    node11173_covers empty_covers (by decide) (by decide) row11177_covers
theorem node11177_eq_data : node11177 = Data.node11177 := by
  have branches : (node11173, Graph.empty) = (Data.node11173, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11173_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11177 row children.1 children.2) row11177_eq_data branches
set_option maxRecDepth 4096 in
theorem row11213_eq_data : row11213 = Data.row11213 := rfl
def node11213 : Graph := .node 11213 row11213 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11213_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11213 :=
  Budget.node_covers _ _ 11213 row11213 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11213_covers
theorem node11213_eq_data : node11213 = Data.node11213 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11213 row children.1 children.2) row11213_eq_data branches
set_option maxRecDepth 4096 in
theorem row11239_eq_data : row11239 = Data.row11239 := rfl
def node11239 : Graph := .node 11239 row11239 node11213 Graph.empty
set_option maxRecDepth 4096 in
theorem node11239_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11239 :=
  Budget.node_covers _ _ 11239 row11239 node11213 Graph.empty
    node11213_covers empty_covers (by decide) (by decide) row11239_covers
theorem node11239_eq_data : node11239 = Data.node11239 := by
  have branches : (node11213, Graph.empty) = (Data.node11213, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11213_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11239 row children.1 children.2) row11239_eq_data branches
set_option maxRecDepth 4096 in
theorem row11197_eq_data : row11197 = Data.row11197 := rfl
def node11197 : Graph := .node 11197 row11197 node11177 node11239
set_option maxRecDepth 4096 in
theorem node11197_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11197 :=
  Budget.node_covers _ _ 11197 row11197 node11177 node11239
    node11177_covers node11239_covers (by decide) (by decide) row11197_covers
theorem node11197_eq_data : node11197 = Data.node11197 := by
  have branches : (node11177, node11239) = (Data.node11177, Data.node11239) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11177_eq_data node11239_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11197 row children.1 children.2) row11197_eq_data branches
set_option maxRecDepth 4096 in
theorem row11171_eq_data : row11171 = Data.row11171 := rfl
def node11171 : Graph := .node 11171 row11171 node11149 node11197
set_option maxRecDepth 4096 in
theorem node11171_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11171 :=
  Budget.node_covers _ _ 11171 row11171 node11149 node11197
    node11149_covers node11197_covers (by decide) (by decide) row11171_covers
theorem node11171_eq_data : node11171 = Data.node11171 := by
  have branches : (node11149, node11197) = (Data.node11149, Data.node11197) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11149_eq_data node11197_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11171 row children.1 children.2) row11171_eq_data branches
set_option maxRecDepth 4096 in
theorem row11117_eq_data : row11117 = Data.row11117 := rfl
def node11117 : Graph := .node 11117 row11117 node11069 node11171
set_option maxRecDepth 4096 in
theorem node11117_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11117 :=
  Budget.node_covers _ _ 11117 row11117 node11069 node11171
    node11069_covers node11171_covers (by decide) (by decide) row11117_covers
theorem node11117_eq_data : node11117 = Data.node11117 := by
  have branches : (node11069, node11171) = (Data.node11069, Data.node11171) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11069_eq_data node11171_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11117 row children.1 children.2) row11117_eq_data branches
set_option maxRecDepth 4096 in
theorem row10993_eq_data : row10993 = Data.row10993 := rfl
def node10993 : Graph := .node 10993 row10993 node10883 node11117
set_option maxRecDepth 4096 in
theorem node10993_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10993 :=
  Budget.node_covers _ _ 10993 row10993 node10883 node11117
    node10883_covers node11117_covers (by decide) (by decide) row10993_covers
theorem node10993_eq_data : node10993 = Data.node10993 := by
  have branches : (node10883, node11117) = (Data.node10883, Data.node11117) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10883_eq_data node11117_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10993 row children.1 children.2) row10993_eq_data branches
set_option maxRecDepth 4096 in
theorem row10753_eq_data : row10753 = Data.row10753 := rfl
def node10753 : Graph := .node 10753 row10753 node10529 node10993
set_option maxRecDepth 4096 in
theorem node10753_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10753 :=
  Budget.node_covers _ _ 10753 row10753 node10529 node10993
    node10529_covers node10993_covers (by decide) (by decide) row10753_covers
theorem node10753_eq_data : node10753 = Data.node10753 := by
  have branches : (node10529, node10993) = (Data.node10529, Data.node10993) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10529_eq_data node10993_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10753 row children.1 children.2) row10753_eq_data branches
set_option maxRecDepth 4096 in
theorem row10289_eq_data : row10289 = Data.row10289 := rfl
def node10289 : Graph := .node 10289 row10289 node9839 node10753
set_option maxRecDepth 4096 in
theorem node10289_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10289 :=
  Budget.node_covers _ _ 10289 row10289 node9839 node10753
    node9839_covers node10753_covers (by decide) (by decide) row10289_covers
theorem node10289_eq_data : node10289 = Data.node10289 := by
  have branches : (node9839, node10753) = (Data.node9839, Data.node10753) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9839_eq_data node10753_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10289 row children.1 children.2) row10289_eq_data branches
set_option maxRecDepth 4096 in
theorem row11251_eq_data : row11251 = Data.row11251 := rfl
def node11251 : Graph := .node 11251 row11251 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11251_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11251 :=
  Budget.node_covers _ _ 11251 row11251 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11251_covers
theorem node11251_eq_data : node11251 = Data.node11251 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11251 row children.1 children.2) row11251_eq_data branches
set_option maxRecDepth 4096 in
theorem row11261_eq_data : row11261 = Data.row11261 := rfl
def node11261 : Graph := .node 11261 row11261 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11261_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11261 :=
  Budget.node_covers _ _ 11261 row11261 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11261_covers
theorem node11261_eq_data : node11261 = Data.node11261 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11261 row children.1 children.2) row11261_eq_data branches
set_option maxRecDepth 4096 in
theorem row11257_eq_data : row11257 = Data.row11257 := rfl
def node11257 : Graph := .node 11257 row11257 node11251 node11261
set_option maxRecDepth 4096 in
theorem node11257_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11257 :=
  Budget.node_covers _ _ 11257 row11257 node11251 node11261
    node11251_covers node11261_covers (by decide) (by decide) row11257_covers
theorem node11257_eq_data : node11257 = Data.node11257 := by
  have branches : (node11251, node11261) = (Data.node11251, Data.node11261) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11251_eq_data node11261_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11257 row children.1 children.2) row11257_eq_data branches
set_option maxRecDepth 4096 in
theorem row11279_eq_data : row11279 = Data.row11279 := rfl
def node11279 : Graph := .node 11279 row11279 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11279_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11279 :=
  Budget.node_covers _ _ 11279 row11279 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11279_covers
theorem node11279_eq_data : node11279 = Data.node11279 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11279 row children.1 children.2) row11279_eq_data branches
set_option maxRecDepth 4096 in
theorem row11287_eq_data : row11287 = Data.row11287 := rfl
def node11287 : Graph := .node 11287 row11287 node11279 Graph.empty
set_option maxRecDepth 4096 in
theorem node11287_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11287 :=
  Budget.node_covers _ _ 11287 row11287 node11279 Graph.empty
    node11279_covers empty_covers (by decide) (by decide) row11287_covers
theorem node11287_eq_data : node11287 = Data.node11287 := by
  have branches : (node11279, Graph.empty) = (Data.node11279, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11279_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11287 row children.1 children.2) row11287_eq_data branches
set_option maxRecDepth 4096 in
theorem row11273_eq_data : row11273 = Data.row11273 := rfl
def node11273 : Graph := .node 11273 row11273 node11257 node11287
set_option maxRecDepth 4096 in
theorem node11273_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11273 :=
  Budget.node_covers _ _ 11273 row11273 node11257 node11287
    node11257_covers node11287_covers (by decide) (by decide) row11273_covers
theorem node11273_eq_data : node11273 = Data.node11273 := by
  have branches : (node11257, node11287) = (Data.node11257, Data.node11287) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11257_eq_data node11287_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11273 row children.1 children.2) row11273_eq_data branches
set_option maxRecDepth 4096 in
theorem row11311_eq_data : row11311 = Data.row11311 := rfl
def node11311 : Graph := .node 11311 row11311 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11311_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11311 :=
  Budget.node_covers _ _ 11311 row11311 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11311_covers
theorem node11311_eq_data : node11311 = Data.node11311 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11311 row children.1 children.2) row11311_eq_data branches
set_option maxRecDepth 4096 in
theorem row11317_eq_data : row11317 = Data.row11317 := rfl
def node11317 : Graph := .node 11317 row11317 node11311 Graph.empty
set_option maxRecDepth 4096 in
theorem node11317_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11317 :=
  Budget.node_covers _ _ 11317 row11317 node11311 Graph.empty
    node11311_covers empty_covers (by decide) (by decide) row11317_covers
theorem node11317_eq_data : node11317 = Data.node11317 := by
  have branches : (node11311, Graph.empty) = (Data.node11311, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11311_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11317 row children.1 children.2) row11317_eq_data branches
set_option maxRecDepth 4096 in
theorem row11329_eq_data : row11329 = Data.row11329 := rfl
def node11329 : Graph := .node 11329 row11329 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11329_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11329 :=
  Budget.node_covers _ _ 11329 row11329 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11329_covers
theorem node11329_eq_data : node11329 = Data.node11329 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11329 row children.1 children.2) row11329_eq_data branches
set_option maxRecDepth 4096 in
theorem row11351_eq_data : row11351 = Data.row11351 := rfl
def node11351 : Graph := .node 11351 row11351 node11329 Graph.empty
set_option maxRecDepth 4096 in
theorem node11351_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11351 :=
  Budget.node_covers _ _ 11351 row11351 node11329 Graph.empty
    node11329_covers empty_covers (by decide) (by decide) row11351_covers
theorem node11351_eq_data : node11351 = Data.node11351 := by
  have branches : (node11329, Graph.empty) = (Data.node11329, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11329_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11351 row children.1 children.2) row11351_eq_data branches
set_option maxRecDepth 4096 in
theorem row11321_eq_data : row11321 = Data.row11321 := rfl
def node11321 : Graph := .node 11321 row11321 node11317 node11351
set_option maxRecDepth 4096 in
theorem node11321_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11321 :=
  Budget.node_covers _ _ 11321 row11321 node11317 node11351
    node11317_covers node11351_covers (by decide) (by decide) row11321_covers
theorem node11321_eq_data : node11321 = Data.node11321 := by
  have branches : (node11317, node11351) = (Data.node11317, Data.node11351) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11317_eq_data node11351_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11321 row children.1 children.2) row11321_eq_data branches
set_option maxRecDepth 4096 in
theorem row11299_eq_data : row11299 = Data.row11299 := rfl
def node11299 : Graph := .node 11299 row11299 node11273 node11321
set_option maxRecDepth 4096 in
theorem node11299_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11299 :=
  Budget.node_covers _ _ 11299 row11299 node11273 node11321
    node11273_covers node11321_covers (by decide) (by decide) row11299_covers
theorem node11299_eq_data : node11299 = Data.node11299 := by
  have branches : (node11273, node11321) = (Data.node11273, Data.node11321) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11273_eq_data node11321_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11299 row children.1 children.2) row11299_eq_data branches
set_option maxRecDepth 4096 in
theorem row11369_eq_data : row11369 = Data.row11369 := rfl
def node11369 : Graph := .node 11369 row11369 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11369_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11369 :=
  Budget.node_covers _ _ 11369 row11369 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11369_covers
theorem node11369_eq_data : node11369 = Data.node11369 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11369 row children.1 children.2) row11369_eq_data branches
set_option maxRecDepth 4096 in
theorem row11383_eq_data : row11383 = Data.row11383 := rfl
def node11383 : Graph := .node 11383 row11383 node11369 Graph.empty
set_option maxRecDepth 4096 in
theorem node11383_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11383 :=
  Budget.node_covers _ _ 11383 row11383 node11369 Graph.empty
    node11369_covers empty_covers (by decide) (by decide) row11383_covers
theorem node11383_eq_data : node11383 = Data.node11383 := by
  have branches : (node11369, Graph.empty) = (Data.node11369, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11369_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11383 row children.1 children.2) row11383_eq_data branches
set_option maxRecDepth 4096 in
theorem row11399_eq_data : row11399 = Data.row11399 := rfl
def node11399 : Graph := .node 11399 row11399 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11399_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11399 :=
  Budget.node_covers _ _ 11399 row11399 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11399_covers
theorem node11399_eq_data : node11399 = Data.node11399 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11399 row children.1 children.2) row11399_eq_data branches
set_option maxRecDepth 4096 in
theorem row11411_eq_data : row11411 = Data.row11411 := rfl
def node11411 : Graph := .node 11411 row11411 node11399 Graph.empty
set_option maxRecDepth 4096 in
theorem node11411_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11411 :=
  Budget.node_covers _ _ 11411 row11411 node11399 Graph.empty
    node11399_covers empty_covers (by decide) (by decide) row11411_covers
theorem node11411_eq_data : node11411 = Data.node11411 := by
  have branches : (node11399, Graph.empty) = (Data.node11399, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11399_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11411 row children.1 children.2) row11411_eq_data branches
set_option maxRecDepth 4096 in
theorem row11393_eq_data : row11393 = Data.row11393 := rfl
def node11393 : Graph := .node 11393 row11393 node11383 node11411
set_option maxRecDepth 4096 in
theorem node11393_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11393 :=
  Budget.node_covers _ _ 11393 row11393 node11383 node11411
    node11383_covers node11411_covers (by decide) (by decide) row11393_covers
theorem node11393_eq_data : node11393 = Data.node11393 := by
  have branches : (node11383, node11411) = (Data.node11383, Data.node11411) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11383_eq_data node11411_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11393 row children.1 children.2) row11393_eq_data branches
set_option maxRecDepth 4096 in
theorem row11437_eq_data : row11437 = Data.row11437 := rfl
def node11437 : Graph := .node 11437 row11437 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11437_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11437 :=
  Budget.node_covers _ _ 11437 row11437 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11437_covers
theorem node11437_eq_data : node11437 = Data.node11437 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11437 row children.1 children.2) row11437_eq_data branches
set_option maxRecDepth 4096 in
theorem row11443_eq_data : row11443 = Data.row11443 := rfl
def node11443 : Graph := .node 11443 row11443 node11437 Graph.empty
set_option maxRecDepth 4096 in
theorem node11443_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11443 :=
  Budget.node_covers _ _ 11443 row11443 node11437 Graph.empty
    node11437_covers empty_covers (by decide) (by decide) row11443_covers
theorem node11443_eq_data : node11443 = Data.node11443 := by
  have branches : (node11437, Graph.empty) = (Data.node11437, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11437_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11443 row children.1 children.2) row11443_eq_data branches
set_option maxRecDepth 4096 in
theorem row11467_eq_data : row11467 = Data.row11467 := rfl
def node11467 : Graph := .node 11467 row11467 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11467_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11467 :=
  Budget.node_covers _ _ 11467 row11467 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11467_covers
theorem node11467_eq_data : node11467 = Data.node11467 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11467 row children.1 children.2) row11467_eq_data branches
set_option maxRecDepth 4096 in
theorem row11471_eq_data : row11471 = Data.row11471 := rfl
def node11471 : Graph := .node 11471 row11471 node11467 Graph.empty
set_option maxRecDepth 4096 in
theorem node11471_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11471 :=
  Budget.node_covers _ _ 11471 row11471 node11467 Graph.empty
    node11467_covers empty_covers (by decide) (by decide) row11471_covers
theorem node11471_eq_data : node11471 = Data.node11471 := by
  have branches : (node11467, Graph.empty) = (Data.node11467, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11467_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11471 row children.1 children.2) row11471_eq_data branches
set_option maxRecDepth 4096 in
theorem row11447_eq_data : row11447 = Data.row11447 := rfl
def node11447 : Graph := .node 11447 row11447 node11443 node11471
set_option maxRecDepth 4096 in
theorem node11447_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11447 :=
  Budget.node_covers _ _ 11447 row11447 node11443 node11471
    node11443_covers node11471_covers (by decide) (by decide) row11447_covers
theorem node11447_eq_data : node11447 = Data.node11447 := by
  have branches : (node11443, node11471) = (Data.node11443, Data.node11471) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11443_eq_data node11471_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11447 row children.1 children.2) row11447_eq_data branches
set_option maxRecDepth 4096 in
theorem row11423_eq_data : row11423 = Data.row11423 := rfl
def node11423 : Graph := .node 11423 row11423 node11393 node11447
set_option maxRecDepth 4096 in
theorem node11423_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11423 :=
  Budget.node_covers _ _ 11423 row11423 node11393 node11447
    node11393_covers node11447_covers (by decide) (by decide) row11423_covers
theorem node11423_eq_data : node11423 = Data.node11423 := by
  have branches : (node11393, node11447) = (Data.node11393, Data.node11447) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11393_eq_data node11447_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11423 row children.1 children.2) row11423_eq_data branches
set_option maxRecDepth 4096 in
theorem row11353_eq_data : row11353 = Data.row11353 := rfl
def node11353 : Graph := .node 11353 row11353 node11299 node11423
set_option maxRecDepth 4096 in
theorem node11353_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11353 :=
  Budget.node_covers _ _ 11353 row11353 node11299 node11423
    node11299_covers node11423_covers (by decide) (by decide) row11353_covers
theorem node11353_eq_data : node11353 = Data.node11353 := by
  have branches : (node11299, node11423) = (Data.node11299, Data.node11423) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11299_eq_data node11423_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11353 row children.1 children.2) row11353_eq_data branches
set_option maxRecDepth 4096 in
theorem row11489_eq_data : row11489 = Data.row11489 := rfl
def node11489 : Graph := .node 11489 row11489 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11489_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11489 :=
  Budget.node_covers _ _ 11489 row11489 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11489_covers
theorem node11489_eq_data : node11489 = Data.node11489 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11489 row children.1 children.2) row11489_eq_data branches
set_option maxRecDepth 4096 in
theorem row11491_eq_data : row11491 = Data.row11491 := rfl
def node11491 : Graph := .node 11491 row11491 node11489 Graph.empty
set_option maxRecDepth 4096 in
theorem node11491_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11491 :=
  Budget.node_covers _ _ 11491 row11491 node11489 Graph.empty
    node11489_covers empty_covers (by decide) (by decide) row11491_covers
theorem node11491_eq_data : node11491 = Data.node11491 := by
  have branches : (node11489, Graph.empty) = (Data.node11489, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11489_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11491 row children.1 children.2) row11491_eq_data branches
set_option maxRecDepth 4096 in
theorem row11503_eq_data : row11503 = Data.row11503 := rfl
def node11503 : Graph := .node 11503 row11503 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11503_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11503 :=
  Budget.node_covers _ _ 11503 row11503 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11503_covers
theorem node11503_eq_data : node11503 = Data.node11503 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11503 row children.1 children.2) row11503_eq_data branches
set_option maxRecDepth 4096 in
theorem row11519_eq_data : row11519 = Data.row11519 := rfl
def node11519 : Graph := .node 11519 row11519 node11503 Graph.empty
set_option maxRecDepth 4096 in
theorem node11519_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11519 :=
  Budget.node_covers _ _ 11519 row11519 node11503 Graph.empty
    node11503_covers empty_covers (by decide) (by decide) row11519_covers
theorem node11519_eq_data : node11519 = Data.node11519 := by
  have branches : (node11503, Graph.empty) = (Data.node11503, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11503_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11519 row children.1 children.2) row11519_eq_data branches
set_option maxRecDepth 4096 in
theorem row11497_eq_data : row11497 = Data.row11497 := rfl
def node11497 : Graph := .node 11497 row11497 node11491 node11519
set_option maxRecDepth 4096 in
theorem node11497_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11497 :=
  Budget.node_covers _ _ 11497 row11497 node11491 node11519
    node11491_covers node11519_covers (by decide) (by decide) row11497_covers
theorem node11497_eq_data : node11497 = Data.node11497 := by
  have branches : (node11491, node11519) = (Data.node11491, Data.node11519) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11491_eq_data node11519_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11497 row children.1 children.2) row11497_eq_data branches
set_option maxRecDepth 4096 in
theorem row11549_eq_data : row11549 = Data.row11549 := rfl
def node11549 : Graph := .node 11549 row11549 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11549_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11549 :=
  Budget.node_covers _ _ 11549 row11549 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11549_covers
theorem node11549_eq_data : node11549 = Data.node11549 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11549 row children.1 children.2) row11549_eq_data branches
set_option maxRecDepth 4096 in
theorem row11551_eq_data : row11551 = Data.row11551 := rfl
def node11551 : Graph := .node 11551 row11551 node11549 Graph.empty
set_option maxRecDepth 4096 in
theorem node11551_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11551 :=
  Budget.node_covers _ _ 11551 row11551 node11549 Graph.empty
    node11549_covers empty_covers (by decide) (by decide) row11551_covers
theorem node11551_eq_data : node11551 = Data.node11551 := by
  have branches : (node11549, Graph.empty) = (Data.node11549, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11549_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11551 row children.1 children.2) row11551_eq_data branches
set_option maxRecDepth 4096 in
theorem row11587_eq_data : row11587 = Data.row11587 := rfl
def node11587 : Graph := .node 11587 row11587 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11587_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11587 :=
  Budget.node_covers _ _ 11587 row11587 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11587_covers
theorem node11587_eq_data : node11587 = Data.node11587 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11587 row children.1 children.2) row11587_eq_data branches
set_option maxRecDepth 4096 in
theorem row11593_eq_data : row11593 = Data.row11593 := rfl
def node11593 : Graph := .node 11593 row11593 node11587 Graph.empty
set_option maxRecDepth 4096 in
theorem node11593_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11593 :=
  Budget.node_covers _ _ 11593 row11593 node11587 Graph.empty
    node11587_covers empty_covers (by decide) (by decide) row11593_covers
theorem node11593_eq_data : node11593 = Data.node11593 := by
  have branches : (node11587, Graph.empty) = (Data.node11587, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11587_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11593 row children.1 children.2) row11593_eq_data branches
set_option maxRecDepth 4096 in
theorem row11579_eq_data : row11579 = Data.row11579 := rfl
def node11579 : Graph := .node 11579 row11579 node11551 node11593
set_option maxRecDepth 4096 in
theorem node11579_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11579 :=
  Budget.node_covers _ _ 11579 row11579 node11551 node11593
    node11551_covers node11593_covers (by decide) (by decide) row11579_covers
theorem node11579_eq_data : node11579 = Data.node11579 := by
  have branches : (node11551, node11593) = (Data.node11551, Data.node11593) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11551_eq_data node11593_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11579 row children.1 children.2) row11579_eq_data branches
set_option maxRecDepth 4096 in
theorem row11527_eq_data : row11527 = Data.row11527 := rfl
def node11527 : Graph := .node 11527 row11527 node11497 node11579
set_option maxRecDepth 4096 in
theorem node11527_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11527 :=
  Budget.node_covers _ _ 11527 row11527 node11497 node11579
    node11497_covers node11579_covers (by decide) (by decide) row11527_covers
theorem node11527_eq_data : node11527 = Data.node11527 := by
  have branches : (node11497, node11579) = (Data.node11497, Data.node11579) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11497_eq_data node11579_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11527 row children.1 children.2) row11527_eq_data branches
set_option maxRecDepth 4096 in
theorem row11617_eq_data : row11617 = Data.row11617 := rfl
def node11617 : Graph := .node 11617 row11617 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11617_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11617 :=
  Budget.node_covers _ _ 11617 row11617 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11617_covers
theorem node11617_eq_data : node11617 = Data.node11617 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11617 row children.1 children.2) row11617_eq_data branches
set_option maxRecDepth 4096 in
theorem row11621_eq_data : row11621 = Data.row11621 := rfl
def node11621 : Graph := .node 11621 row11621 node11617 Graph.empty
set_option maxRecDepth 4096 in
theorem node11621_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11621 :=
  Budget.node_covers _ _ 11621 row11621 node11617 Graph.empty
    node11617_covers empty_covers (by decide) (by decide) row11621_covers
theorem node11621_eq_data : node11621 = Data.node11621 := by
  have branches : (node11617, Graph.empty) = (Data.node11617, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11617_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11621 row children.1 children.2) row11621_eq_data branches
set_option maxRecDepth 4096 in
theorem row11657_eq_data : row11657 = Data.row11657 := rfl
def node11657 : Graph := .node 11657 row11657 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11657_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11657 :=
  Budget.node_covers _ _ 11657 row11657 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11657_covers
theorem node11657_eq_data : node11657 = Data.node11657 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11657 row children.1 children.2) row11657_eq_data branches
set_option maxRecDepth 4096 in
theorem row11677_eq_data : row11677 = Data.row11677 := rfl
def node11677 : Graph := .node 11677 row11677 node11657 Graph.empty
set_option maxRecDepth 4096 in
theorem node11677_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11677 :=
  Budget.node_covers _ _ 11677 row11677 node11657 Graph.empty
    node11657_covers empty_covers (by decide) (by decide) row11677_covers
theorem node11677_eq_data : node11677 = Data.node11677 := by
  have branches : (node11657, Graph.empty) = (Data.node11657, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11657_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11677 row children.1 children.2) row11677_eq_data branches
set_option maxRecDepth 4096 in
theorem row11633_eq_data : row11633 = Data.row11633 := rfl
def node11633 : Graph := .node 11633 row11633 node11621 node11677
set_option maxRecDepth 4096 in
theorem node11633_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11633 :=
  Budget.node_covers _ _ 11633 row11633 node11621 node11677
    node11621_covers node11677_covers (by decide) (by decide) row11633_covers
theorem node11633_eq_data : node11633 = Data.node11633 := by
  have branches : (node11621, node11677) = (Data.node11621, Data.node11677) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11621_eq_data node11677_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11633 row children.1 children.2) row11633_eq_data branches
set_option maxRecDepth 4096 in
theorem row11689_eq_data : row11689 = Data.row11689 := rfl
def node11689 : Graph := .node 11689 row11689 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11689 :=
  Budget.node_covers _ _ 11689 row11689 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11689_covers
theorem node11689_eq_data : node11689 = Data.node11689 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11689 row children.1 children.2) row11689_eq_data branches
set_option maxRecDepth 4096 in
theorem row11699_eq_data : row11699 = Data.row11699 := rfl
def node11699 : Graph := .node 11699 row11699 node11689 Graph.empty
set_option maxRecDepth 4096 in
theorem node11699_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11699 :=
  Budget.node_covers _ _ 11699 row11699 node11689 Graph.empty
    node11689_covers empty_covers (by decide) (by decide) row11699_covers
theorem node11699_eq_data : node11699 = Data.node11699 := by
  have branches : (node11689, Graph.empty) = (Data.node11689, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11689_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11699 row children.1 children.2) row11699_eq_data branches
set_option maxRecDepth 4096 in
theorem row11717_eq_data : row11717 = Data.row11717 := rfl
def node11717 : Graph := .node 11717 row11717 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11717_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11717 :=
  Budget.node_covers _ _ 11717 row11717 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11717_covers
theorem node11717_eq_data : node11717 = Data.node11717 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11717 row children.1 children.2) row11717_eq_data branches
set_option maxRecDepth 4096 in
theorem row11719_eq_data : row11719 = Data.row11719 := rfl
def node11719 : Graph := .node 11719 row11719 node11717 Graph.empty
set_option maxRecDepth 4096 in
theorem node11719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11719 :=
  Budget.node_covers _ _ 11719 row11719 node11717 Graph.empty
    node11717_covers empty_covers (by decide) (by decide) row11719_covers
theorem node11719_eq_data : node11719 = Data.node11719 := by
  have branches : (node11717, Graph.empty) = (Data.node11717, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11717_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11719 row children.1 children.2) row11719_eq_data branches
set_option maxRecDepth 4096 in
theorem row11701_eq_data : row11701 = Data.row11701 := rfl
def node11701 : Graph := .node 11701 row11701 node11699 node11719
set_option maxRecDepth 4096 in
theorem node11701_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11701 :=
  Budget.node_covers _ _ 11701 row11701 node11699 node11719
    node11699_covers node11719_covers (by decide) (by decide) row11701_covers
theorem node11701_eq_data : node11701 = Data.node11701 := by
  have branches : (node11699, node11719) = (Data.node11699, Data.node11719) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11699_eq_data node11719_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11701 row children.1 children.2) row11701_eq_data branches
set_option maxRecDepth 4096 in
theorem row11681_eq_data : row11681 = Data.row11681 := rfl
def node11681 : Graph := .node 11681 row11681 node11633 node11701
set_option maxRecDepth 4096 in
theorem node11681_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11681 :=
  Budget.node_covers _ _ 11681 row11681 node11633 node11701
    node11633_covers node11701_covers (by decide) (by decide) row11681_covers
theorem node11681_eq_data : node11681 = Data.node11681 := by
  have branches : (node11633, node11701) = (Data.node11633, Data.node11701) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11633_eq_data node11701_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11681 row children.1 children.2) row11681_eq_data branches
set_option maxRecDepth 4096 in
theorem row11597_eq_data : row11597 = Data.row11597 := rfl
def node11597 : Graph := .node 11597 row11597 node11527 node11681
set_option maxRecDepth 4096 in
theorem node11597_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11597 :=
  Budget.node_covers _ _ 11597 row11597 node11527 node11681
    node11527_covers node11681_covers (by decide) (by decide) row11597_covers
theorem node11597_eq_data : node11597 = Data.node11597 := by
  have branches : (node11527, node11681) = (Data.node11527, Data.node11681) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11527_eq_data node11681_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11597 row children.1 children.2) row11597_eq_data branches
set_option maxRecDepth 4096 in
theorem row11483_eq_data : row11483 = Data.row11483 := rfl
def node11483 : Graph := .node 11483 row11483 node11353 node11597
set_option maxRecDepth 4096 in
theorem node11483_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11483 :=
  Budget.node_covers _ _ 11483 row11483 node11353 node11597
    node11353_covers node11597_covers (by decide) (by decide) row11483_covers
theorem node11483_eq_data : node11483 = Data.node11483 := by
  have branches : (node11353, node11597) = (Data.node11353, Data.node11597) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11353_eq_data node11597_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11483 row children.1 children.2) row11483_eq_data branches
set_option maxRecDepth 4096 in
theorem row11743_eq_data : row11743 = Data.row11743 := rfl
def node11743 : Graph := .node 11743 row11743 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11743_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11743 :=
  Budget.node_covers _ _ 11743 row11743 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11743_covers
theorem node11743_eq_data : node11743 = Data.node11743 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11743 row children.1 children.2) row11743_eq_data branches
set_option maxRecDepth 4096 in
theorem row11777_eq_data : row11777 = Data.row11777 := rfl
def node11777 : Graph := .node 11777 row11777 node11743 Graph.empty
set_option maxRecDepth 4096 in
theorem node11777_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11777 :=
  Budget.node_covers _ _ 11777 row11777 node11743 Graph.empty
    node11743_covers empty_covers (by decide) (by decide) row11777_covers
theorem node11777_eq_data : node11777 = Data.node11777 := by
  have branches : (node11743, Graph.empty) = (Data.node11743, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11743_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11777 row children.1 children.2) row11777_eq_data branches
set_option maxRecDepth 4096 in
theorem row11783_eq_data : row11783 = Data.row11783 := rfl
def node11783 : Graph := .node 11783 row11783 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11783_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11783 :=
  Budget.node_covers _ _ 11783 row11783 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11783_covers
theorem node11783_eq_data : node11783 = Data.node11783 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11783 row children.1 children.2) row11783_eq_data branches
set_option maxRecDepth 4096 in
theorem row11789_eq_data : row11789 = Data.row11789 := rfl
def node11789 : Graph := .node 11789 row11789 node11783 Graph.empty
set_option maxRecDepth 4096 in
theorem node11789_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11789 :=
  Budget.node_covers _ _ 11789 row11789 node11783 Graph.empty
    node11783_covers empty_covers (by decide) (by decide) row11789_covers
theorem node11789_eq_data : node11789 = Data.node11789 := by
  have branches : (node11783, Graph.empty) = (Data.node11783, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11783_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11789 row children.1 children.2) row11789_eq_data branches
set_option maxRecDepth 4096 in
theorem row11779_eq_data : row11779 = Data.row11779 := rfl
def node11779 : Graph := .node 11779 row11779 node11777 node11789
set_option maxRecDepth 4096 in
theorem node11779_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11779 :=
  Budget.node_covers _ _ 11779 row11779 node11777 node11789
    node11777_covers node11789_covers (by decide) (by decide) row11779_covers
theorem node11779_eq_data : node11779 = Data.node11779 := by
  have branches : (node11777, node11789) = (Data.node11777, Data.node11789) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11777_eq_data node11789_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11779 row children.1 children.2) row11779_eq_data branches
set_option maxRecDepth 4096 in
theorem row11807_eq_data : row11807 = Data.row11807 := rfl
def node11807 : Graph := .node 11807 row11807 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11807_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11807 :=
  Budget.node_covers _ _ 11807 row11807 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11807_covers
theorem node11807_eq_data : node11807 = Data.node11807 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11807 row children.1 children.2) row11807_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
