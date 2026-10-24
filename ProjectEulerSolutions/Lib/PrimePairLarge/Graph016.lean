import ProjectEulerSolutions.Lib.PrimePairLarge.Graph015
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row8209_eq_data : row8209 = Data.row8209 := rfl
def node8209 : Graph := .node 8209 row8209 node8191 node8221
set_option maxRecDepth 4096 in
theorem node8209_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8209 :=
  Budget.node_covers _ _ 8209 row8209 node8191 node8221
    node8191_covers node8221_covers (by decide) (by decide) row8209_covers
theorem node8209_eq_data : node8209 = Data.node8209 := by
  have branches : (node8191, node8221) = (Data.node8191, Data.node8221) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8191_eq_data node8221_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8209 row children.1 children.2) row8209_eq_data branches
set_option maxRecDepth 4096 in
theorem row8171_eq_data : row8171 = Data.row8171 := rfl
def node8171 : Graph := .node 8171 row8171 node8147 node8209
set_option maxRecDepth 4096 in
theorem node8171_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8171 :=
  Budget.node_covers _ _ 8171 row8171 node8147 node8209
    node8147_covers node8209_covers (by decide) (by decide) row8171_covers
theorem node8171_eq_data : node8171 = Data.node8171 := by
  have branches : (node8147, node8209) = (Data.node8147, Data.node8209) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8147_eq_data node8209_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8171 row children.1 children.2) row8171_eq_data branches
set_option maxRecDepth 4096 in
theorem row8233_eq_data : row8233 = Data.row8233 := rfl
def node8233 : Graph := .node 8233 row8233 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8233_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8233 :=
  Budget.node_covers _ _ 8233 row8233 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8233_covers
theorem node8233_eq_data : node8233 = Data.node8233 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8233 row children.1 children.2) row8233_eq_data branches
set_option maxRecDepth 4096 in
theorem row8237_eq_data : row8237 = Data.row8237 := rfl
def node8237 : Graph := .node 8237 row8237 node8233 Graph.empty
set_option maxRecDepth 4096 in
theorem node8237_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8237 :=
  Budget.node_covers _ _ 8237 row8237 node8233 Graph.empty
    node8233_covers empty_covers (by decide) (by decide) row8237_covers
theorem node8237_eq_data : node8237 = Data.node8237 := by
  have branches : (node8233, Graph.empty) = (Data.node8233, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8233_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8237 row children.1 children.2) row8237_eq_data branches
set_option maxRecDepth 4096 in
theorem row8263_eq_data : row8263 = Data.row8263 := rfl
def node8263 : Graph := .node 8263 row8263 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8263_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8263 :=
  Budget.node_covers _ _ 8263 row8263 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8263_covers
theorem node8263_eq_data : node8263 = Data.node8263 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8263 row children.1 children.2) row8263_eq_data branches
set_option maxRecDepth 4096 in
theorem row8269_eq_data : row8269 = Data.row8269 := rfl
def node8269 : Graph := .node 8269 row8269 node8263 Graph.empty
set_option maxRecDepth 4096 in
theorem node8269_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8269 :=
  Budget.node_covers _ _ 8269 row8269 node8263 Graph.empty
    node8263_covers empty_covers (by decide) (by decide) row8269_covers
theorem node8269_eq_data : node8269 = Data.node8269 := by
  have branches : (node8263, Graph.empty) = (Data.node8263, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8263_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8269 row children.1 children.2) row8269_eq_data branches
set_option maxRecDepth 4096 in
theorem row8243_eq_data : row8243 = Data.row8243 := rfl
def node8243 : Graph := .node 8243 row8243 node8237 node8269
set_option maxRecDepth 4096 in
theorem node8243_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8243 :=
  Budget.node_covers _ _ 8243 row8243 node8237 node8269
    node8237_covers node8269_covers (by decide) (by decide) row8243_covers
theorem node8243_eq_data : node8243 = Data.node8243 := by
  have branches : (node8237, node8269) = (Data.node8237, Data.node8269) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8237_eq_data node8269_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8243 row children.1 children.2) row8243_eq_data branches
set_option maxRecDepth 4096 in
theorem row8287_eq_data : row8287 = Data.row8287 := rfl
def node8287 : Graph := .node 8287 row8287 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8287_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8287 :=
  Budget.node_covers _ _ 8287 row8287 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8287_covers
theorem node8287_eq_data : node8287 = Data.node8287 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8287 row children.1 children.2) row8287_eq_data branches
set_option maxRecDepth 4096 in
theorem row8291_eq_data : row8291 = Data.row8291 := rfl
def node8291 : Graph := .node 8291 row8291 node8287 Graph.empty
set_option maxRecDepth 4096 in
theorem node8291_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8291 :=
  Budget.node_covers _ _ 8291 row8291 node8287 Graph.empty
    node8287_covers empty_covers (by decide) (by decide) row8291_covers
theorem node8291_eq_data : node8291 = Data.node8291 := by
  have branches : (node8287, Graph.empty) = (Data.node8287, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8287_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8291 row children.1 children.2) row8291_eq_data branches
set_option maxRecDepth 4096 in
theorem row8297_eq_data : row8297 = Data.row8297 := rfl
def node8297 : Graph := .node 8297 row8297 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8297_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8297 :=
  Budget.node_covers _ _ 8297 row8297 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8297_covers
theorem node8297_eq_data : node8297 = Data.node8297 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8297 row children.1 children.2) row8297_eq_data branches
set_option maxRecDepth 4096 in
theorem row8311_eq_data : row8311 = Data.row8311 := rfl
def node8311 : Graph := .node 8311 row8311 node8297 Graph.empty
set_option maxRecDepth 4096 in
theorem node8311_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8311 :=
  Budget.node_covers _ _ 8311 row8311 node8297 Graph.empty
    node8297_covers empty_covers (by decide) (by decide) row8311_covers
theorem node8311_eq_data : node8311 = Data.node8311 := by
  have branches : (node8297, Graph.empty) = (Data.node8297, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8297_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8311 row children.1 children.2) row8311_eq_data branches
set_option maxRecDepth 4096 in
theorem row8293_eq_data : row8293 = Data.row8293 := rfl
def node8293 : Graph := .node 8293 row8293 node8291 node8311
set_option maxRecDepth 4096 in
theorem node8293_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8293 :=
  Budget.node_covers _ _ 8293 row8293 node8291 node8311
    node8291_covers node8311_covers (by decide) (by decide) row8293_covers
theorem node8293_eq_data : node8293 = Data.node8293 := by
  have branches : (node8291, node8311) = (Data.node8291, Data.node8311) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8291_eq_data node8311_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8293 row children.1 children.2) row8293_eq_data branches
set_option maxRecDepth 4096 in
theorem row8273_eq_data : row8273 = Data.row8273 := rfl
def node8273 : Graph := .node 8273 row8273 node8243 node8293
set_option maxRecDepth 4096 in
theorem node8273_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8273 :=
  Budget.node_covers _ _ 8273 row8273 node8243 node8293
    node8243_covers node8293_covers (by decide) (by decide) row8273_covers
theorem node8273_eq_data : node8273 = Data.node8273 := by
  have branches : (node8243, node8293) = (Data.node8243, Data.node8293) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8243_eq_data node8293_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8273 row children.1 children.2) row8273_eq_data branches
set_option maxRecDepth 4096 in
theorem row8231_eq_data : row8231 = Data.row8231 := rfl
def node8231 : Graph := .node 8231 row8231 node8171 node8273
set_option maxRecDepth 4096 in
theorem node8231_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8231 :=
  Budget.node_covers _ _ 8231 row8231 node8171 node8273
    node8171_covers node8273_covers (by decide) (by decide) row8231_covers
theorem node8231_eq_data : node8231 = Data.node8231 := by
  have branches : (node8171, node8273) = (Data.node8171, Data.node8273) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8171_eq_data node8273_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8231 row children.1 children.2) row8231_eq_data branches
set_option maxRecDepth 4096 in
theorem row8329_eq_data : row8329 = Data.row8329 := rfl
def node8329 : Graph := .node 8329 row8329 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8329_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8329 :=
  Budget.node_covers _ _ 8329 row8329 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8329_covers
theorem node8329_eq_data : node8329 = Data.node8329 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8329 row children.1 children.2) row8329_eq_data branches
set_option maxRecDepth 4096 in
theorem row8353_eq_data : row8353 = Data.row8353 := rfl
def node8353 : Graph := .node 8353 row8353 node8329 Graph.empty
set_option maxRecDepth 4096 in
theorem node8353_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8353 :=
  Budget.node_covers _ _ 8353 row8353 node8329 Graph.empty
    node8329_covers empty_covers (by decide) (by decide) row8353_covers
theorem node8353_eq_data : node8353 = Data.node8353 := by
  have branches : (node8329, Graph.empty) = (Data.node8329, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8329_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8353 row children.1 children.2) row8353_eq_data branches
set_option maxRecDepth 4096 in
theorem row8369_eq_data : row8369 = Data.row8369 := rfl
def node8369 : Graph := .node 8369 row8369 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8369_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8369 :=
  Budget.node_covers _ _ 8369 row8369 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8369_covers
theorem node8369_eq_data : node8369 = Data.node8369 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8369 row children.1 children.2) row8369_eq_data branches
set_option maxRecDepth 4096 in
theorem row8377_eq_data : row8377 = Data.row8377 := rfl
def node8377 : Graph := .node 8377 row8377 node8369 Graph.empty
set_option maxRecDepth 4096 in
theorem node8377_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8377 :=
  Budget.node_covers _ _ 8377 row8377 node8369 Graph.empty
    node8369_covers empty_covers (by decide) (by decide) row8377_covers
theorem node8377_eq_data : node8377 = Data.node8377 := by
  have branches : (node8369, Graph.empty) = (Data.node8369, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8369_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8377 row children.1 children.2) row8377_eq_data branches
set_option maxRecDepth 4096 in
theorem row8363_eq_data : row8363 = Data.row8363 := rfl
def node8363 : Graph := .node 8363 row8363 node8353 node8377
set_option maxRecDepth 4096 in
theorem node8363_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8363 :=
  Budget.node_covers _ _ 8363 row8363 node8353 node8377
    node8353_covers node8377_covers (by decide) (by decide) row8363_covers
theorem node8363_eq_data : node8363 = Data.node8363 := by
  have branches : (node8353, node8377) = (Data.node8353, Data.node8377) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8353_eq_data node8377_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8363 row children.1 children.2) row8363_eq_data branches
set_option maxRecDepth 4096 in
theorem row8389_eq_data : row8389 = Data.row8389 := rfl
def node8389 : Graph := .node 8389 row8389 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8389_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8389 :=
  Budget.node_covers _ _ 8389 row8389 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8389_covers
theorem node8389_eq_data : node8389 = Data.node8389 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8389 row children.1 children.2) row8389_eq_data branches
set_option maxRecDepth 4096 in
theorem row8419_eq_data : row8419 = Data.row8419 := rfl
def node8419 : Graph := .node 8419 row8419 node8389 Graph.empty
set_option maxRecDepth 4096 in
theorem node8419_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8419 :=
  Budget.node_covers _ _ 8419 row8419 node8389 Graph.empty
    node8389_covers empty_covers (by decide) (by decide) row8419_covers
theorem node8419_eq_data : node8419 = Data.node8419 := by
  have branches : (node8389, Graph.empty) = (Data.node8389, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8389_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8419 row children.1 children.2) row8419_eq_data branches
set_option maxRecDepth 4096 in
theorem row8429_eq_data : row8429 = Data.row8429 := rfl
def node8429 : Graph := .node 8429 row8429 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8429_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8429 :=
  Budget.node_covers _ _ 8429 row8429 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8429_covers
theorem node8429_eq_data : node8429 = Data.node8429 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8429 row children.1 children.2) row8429_eq_data branches
set_option maxRecDepth 4096 in
theorem row8431_eq_data : row8431 = Data.row8431 := rfl
def node8431 : Graph := .node 8431 row8431 node8429 Graph.empty
set_option maxRecDepth 4096 in
theorem node8431_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8431 :=
  Budget.node_covers _ _ 8431 row8431 node8429 Graph.empty
    node8429_covers empty_covers (by decide) (by decide) row8431_covers
theorem node8431_eq_data : node8431 = Data.node8431 := by
  have branches : (node8429, Graph.empty) = (Data.node8429, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8429_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8431 row children.1 children.2) row8431_eq_data branches
set_option maxRecDepth 4096 in
theorem row8423_eq_data : row8423 = Data.row8423 := rfl
def node8423 : Graph := .node 8423 row8423 node8419 node8431
set_option maxRecDepth 4096 in
theorem node8423_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8423 :=
  Budget.node_covers _ _ 8423 row8423 node8419 node8431
    node8419_covers node8431_covers (by decide) (by decide) row8423_covers
theorem node8423_eq_data : node8423 = Data.node8423 := by
  have branches : (node8419, node8431) = (Data.node8419, Data.node8431) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8419_eq_data node8431_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8423 row children.1 children.2) row8423_eq_data branches
set_option maxRecDepth 4096 in
theorem row8387_eq_data : row8387 = Data.row8387 := rfl
def node8387 : Graph := .node 8387 row8387 node8363 node8423
set_option maxRecDepth 4096 in
theorem node8387_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8387 :=
  Budget.node_covers _ _ 8387 row8387 node8363 node8423
    node8363_covers node8423_covers (by decide) (by decide) row8387_covers
theorem node8387_eq_data : node8387 = Data.node8387 := by
  have branches : (node8363, node8423) = (Data.node8363, Data.node8423) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8363_eq_data node8423_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8387 row children.1 children.2) row8387_eq_data branches
set_option maxRecDepth 4096 in
theorem row8447_eq_data : row8447 = Data.row8447 := rfl
def node8447 : Graph := .node 8447 row8447 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8447_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8447 :=
  Budget.node_covers _ _ 8447 row8447 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8447_covers
theorem node8447_eq_data : node8447 = Data.node8447 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8447 row children.1 children.2) row8447_eq_data branches
set_option maxRecDepth 4096 in
theorem row8461_eq_data : row8461 = Data.row8461 := rfl
def node8461 : Graph := .node 8461 row8461 node8447 Graph.empty
set_option maxRecDepth 4096 in
theorem node8461_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8461 :=
  Budget.node_covers _ _ 8461 row8461 node8447 Graph.empty
    node8447_covers empty_covers (by decide) (by decide) row8461_covers
theorem node8461_eq_data : node8461 = Data.node8461 := by
  have branches : (node8447, Graph.empty) = (Data.node8447, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8447_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8461 row children.1 children.2) row8461_eq_data branches
set_option maxRecDepth 4096 in
theorem row8501_eq_data : row8501 = Data.row8501 := rfl
def node8501 : Graph := .node 8501 row8501 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8501_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8501 :=
  Budget.node_covers _ _ 8501 row8501 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8501_covers
theorem node8501_eq_data : node8501 = Data.node8501 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8501 row children.1 children.2) row8501_eq_data branches
set_option maxRecDepth 4096 in
theorem row8513_eq_data : row8513 = Data.row8513 := rfl
def node8513 : Graph := .node 8513 row8513 node8501 Graph.empty
set_option maxRecDepth 4096 in
theorem node8513_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8513 :=
  Budget.node_covers _ _ 8513 row8513 node8501 Graph.empty
    node8501_covers empty_covers (by decide) (by decide) row8513_covers
theorem node8513_eq_data : node8513 = Data.node8513 := by
  have branches : (node8501, Graph.empty) = (Data.node8501, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8501_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8513 row children.1 children.2) row8513_eq_data branches
set_option maxRecDepth 4096 in
theorem row8467_eq_data : row8467 = Data.row8467 := rfl
def node8467 : Graph := .node 8467 row8467 node8461 node8513
set_option maxRecDepth 4096 in
theorem node8467_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8467 :=
  Budget.node_covers _ _ 8467 row8467 node8461 node8513
    node8461_covers node8513_covers (by decide) (by decide) row8467_covers
theorem node8467_eq_data : node8467 = Data.node8467 := by
  have branches : (node8461, node8513) = (Data.node8461, Data.node8513) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8461_eq_data node8513_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8467 row children.1 children.2) row8467_eq_data branches
set_option maxRecDepth 4096 in
theorem row8527_eq_data : row8527 = Data.row8527 := rfl
def node8527 : Graph := .node 8527 row8527 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8527_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8527 :=
  Budget.node_covers _ _ 8527 row8527 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8527_covers
theorem node8527_eq_data : node8527 = Data.node8527 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8527 row children.1 children.2) row8527_eq_data branches
set_option maxRecDepth 4096 in
theorem row8537_eq_data : row8537 = Data.row8537 := rfl
def node8537 : Graph := .node 8537 row8537 node8527 Graph.empty
set_option maxRecDepth 4096 in
theorem node8537_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8537 :=
  Budget.node_covers _ _ 8537 row8537 node8527 Graph.empty
    node8527_covers empty_covers (by decide) (by decide) row8537_covers
theorem node8537_eq_data : node8537 = Data.node8537 := by
  have branches : (node8527, Graph.empty) = (Data.node8527, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8527_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8537 row children.1 children.2) row8537_eq_data branches
set_option maxRecDepth 4096 in
theorem row8543_eq_data : row8543 = Data.row8543 := rfl
def node8543 : Graph := .node 8543 row8543 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8543_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8543 :=
  Budget.node_covers _ _ 8543 row8543 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8543_covers
theorem node8543_eq_data : node8543 = Data.node8543 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8543 row children.1 children.2) row8543_eq_data branches
set_option maxRecDepth 4096 in
theorem row8563_eq_data : row8563 = Data.row8563 := rfl
def node8563 : Graph := .node 8563 row8563 node8543 Graph.empty
set_option maxRecDepth 4096 in
theorem node8563_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8563 :=
  Budget.node_covers _ _ 8563 row8563 node8543 Graph.empty
    node8543_covers empty_covers (by decide) (by decide) row8563_covers
theorem node8563_eq_data : node8563 = Data.node8563 := by
  have branches : (node8543, Graph.empty) = (Data.node8543, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8543_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8563 row children.1 children.2) row8563_eq_data branches
set_option maxRecDepth 4096 in
theorem row8539_eq_data : row8539 = Data.row8539 := rfl
def node8539 : Graph := .node 8539 row8539 node8537 node8563
set_option maxRecDepth 4096 in
theorem node8539_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8539 :=
  Budget.node_covers _ _ 8539 row8539 node8537 node8563
    node8537_covers node8563_covers (by decide) (by decide) row8539_covers
theorem node8539_eq_data : node8539 = Data.node8539 := by
  have branches : (node8537, node8563) = (Data.node8537, Data.node8563) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8537_eq_data node8563_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8539 row children.1 children.2) row8539_eq_data branches
set_option maxRecDepth 4096 in
theorem row8521_eq_data : row8521 = Data.row8521 := rfl
def node8521 : Graph := .node 8521 row8521 node8467 node8539
set_option maxRecDepth 4096 in
theorem node8521_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8521 :=
  Budget.node_covers _ _ 8521 row8521 node8467 node8539
    node8467_covers node8539_covers (by decide) (by decide) row8521_covers
theorem node8521_eq_data : node8521 = Data.node8521 := by
  have branches : (node8467, node8539) = (Data.node8467, Data.node8539) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8467_eq_data node8539_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8521 row children.1 children.2) row8521_eq_data branches
set_option maxRecDepth 4096 in
theorem row8443_eq_data : row8443 = Data.row8443 := rfl
def node8443 : Graph := .node 8443 row8443 node8387 node8521
set_option maxRecDepth 4096 in
theorem node8443_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8443 :=
  Budget.node_covers _ _ 8443 row8443 node8387 node8521
    node8387_covers node8521_covers (by decide) (by decide) row8443_covers
theorem node8443_eq_data : node8443 = Data.node8443 := by
  have branches : (node8387, node8521) = (Data.node8387, Data.node8521) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8387_eq_data node8521_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8443 row children.1 children.2) row8443_eq_data branches
set_option maxRecDepth 4096 in
theorem row8317_eq_data : row8317 = Data.row8317 := rfl
def node8317 : Graph := .node 8317 row8317 node8231 node8443
set_option maxRecDepth 4096 in
theorem node8317_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8317 :=
  Budget.node_covers _ _ 8317 row8317 node8231 node8443
    node8231_covers node8443_covers (by decide) (by decide) row8317_covers
theorem node8317_eq_data : node8317 = Data.node8317 := by
  have branches : (node8231, node8443) = (Data.node8231, Data.node8443) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8231_eq_data node8443_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8317 row children.1 children.2) row8317_eq_data branches
set_option maxRecDepth 4096 in
theorem row8111_eq_data : row8111 = Data.row8111 := rfl
def node8111 : Graph := .node 8111 row8111 node7879 node8317
set_option maxRecDepth 4096 in
theorem node8111_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8111 :=
  Budget.node_covers _ _ 8111 row8111 node7879 node8317
    node7879_covers node8317_covers (by decide) (by decide) row8111_covers
theorem node8111_eq_data : node8111 = Data.node8111 := by
  have branches : (node7879, node8317) = (Data.node7879, Data.node8317) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node7879_eq_data node8317_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8111 row children.1 children.2) row8111_eq_data branches
set_option maxRecDepth 4096 in
theorem row8581_eq_data : row8581 = Data.row8581 := rfl
def node8581 : Graph := .node 8581 row8581 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8581_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8581 :=
  Budget.node_covers _ _ 8581 row8581 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8581_covers
theorem node8581_eq_data : node8581 = Data.node8581 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8581 row children.1 children.2) row8581_eq_data branches
set_option maxRecDepth 4096 in
theorem row8599_eq_data : row8599 = Data.row8599 := rfl
def node8599 : Graph := .node 8599 row8599 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8599_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8599 :=
  Budget.node_covers _ _ 8599 row8599 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8599_covers
theorem node8599_eq_data : node8599 = Data.node8599 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8599 row children.1 children.2) row8599_eq_data branches
set_option maxRecDepth 4096 in
theorem row8597_eq_data : row8597 = Data.row8597 := rfl
def node8597 : Graph := .node 8597 row8597 node8581 node8599
set_option maxRecDepth 4096 in
theorem node8597_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8597 :=
  Budget.node_covers _ _ 8597 row8597 node8581 node8599
    node8581_covers node8599_covers (by decide) (by decide) row8597_covers
theorem node8597_eq_data : node8597 = Data.node8597 := by
  have branches : (node8581, node8599) = (Data.node8581, Data.node8599) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8581_eq_data node8599_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8597 row children.1 children.2) row8597_eq_data branches
set_option maxRecDepth 4096 in
theorem row8623_eq_data : row8623 = Data.row8623 := rfl
def node8623 : Graph := .node 8623 row8623 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8623_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8623 :=
  Budget.node_covers _ _ 8623 row8623 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8623_covers
theorem node8623_eq_data : node8623 = Data.node8623 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8623 row children.1 children.2) row8623_eq_data branches
set_option maxRecDepth 4096 in
theorem row8627_eq_data : row8627 = Data.row8627 := rfl
def node8627 : Graph := .node 8627 row8627 node8623 Graph.empty
set_option maxRecDepth 4096 in
theorem node8627_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8627 :=
  Budget.node_covers _ _ 8627 row8627 node8623 Graph.empty
    node8623_covers empty_covers (by decide) (by decide) row8627_covers
theorem node8627_eq_data : node8627 = Data.node8627 := by
  have branches : (node8623, Graph.empty) = (Data.node8623, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8623_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8627 row children.1 children.2) row8627_eq_data branches
set_option maxRecDepth 4096 in
theorem row8609_eq_data : row8609 = Data.row8609 := rfl
def node8609 : Graph := .node 8609 row8609 node8597 node8627
set_option maxRecDepth 4096 in
theorem node8609_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8609 :=
  Budget.node_covers _ _ 8609 row8609 node8597 node8627
    node8597_covers node8627_covers (by decide) (by decide) row8609_covers
theorem node8609_eq_data : node8609 = Data.node8609 := by
  have branches : (node8597, node8627) = (Data.node8597, Data.node8627) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8597_eq_data node8627_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8609 row children.1 children.2) row8609_eq_data branches
set_option maxRecDepth 4096 in
theorem row8641_eq_data : row8641 = Data.row8641 := rfl
def node8641 : Graph := .node 8641 row8641 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8641_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8641 :=
  Budget.node_covers _ _ 8641 row8641 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8641_covers
theorem node8641_eq_data : node8641 = Data.node8641 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8641 row children.1 children.2) row8641_eq_data branches
set_option maxRecDepth 4096 in
theorem row8647_eq_data : row8647 = Data.row8647 := rfl
def node8647 : Graph := .node 8647 row8647 node8641 Graph.empty
set_option maxRecDepth 4096 in
theorem node8647_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8647 :=
  Budget.node_covers _ _ 8647 row8647 node8641 Graph.empty
    node8641_covers empty_covers (by decide) (by decide) row8647_covers
theorem node8647_eq_data : node8647 = Data.node8647 := by
  have branches : (node8641, Graph.empty) = (Data.node8641, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8641_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8647 row children.1 children.2) row8647_eq_data branches
set_option maxRecDepth 4096 in
theorem row8669_eq_data : row8669 = Data.row8669 := rfl
def node8669 : Graph := .node 8669 row8669 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8669_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8669 :=
  Budget.node_covers _ _ 8669 row8669 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8669_covers
theorem node8669_eq_data : node8669 = Data.node8669 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8669 row children.1 children.2) row8669_eq_data branches
set_option maxRecDepth 4096 in
theorem row8677_eq_data : row8677 = Data.row8677 := rfl
def node8677 : Graph := .node 8677 row8677 node8669 Graph.empty
set_option maxRecDepth 4096 in
theorem node8677_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8677 :=
  Budget.node_covers _ _ 8677 row8677 node8669 Graph.empty
    node8669_covers empty_covers (by decide) (by decide) row8677_covers
theorem node8677_eq_data : node8677 = Data.node8677 := by
  have branches : (node8669, Graph.empty) = (Data.node8669, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8669_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8677 row children.1 children.2) row8677_eq_data branches
set_option maxRecDepth 4096 in
theorem row8663_eq_data : row8663 = Data.row8663 := rfl
def node8663 : Graph := .node 8663 row8663 node8647 node8677
set_option maxRecDepth 4096 in
theorem node8663_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8663 :=
  Budget.node_covers _ _ 8663 row8663 node8647 node8677
    node8647_covers node8677_covers (by decide) (by decide) row8663_covers
theorem node8663_eq_data : node8663 = Data.node8663 := by
  have branches : (node8647, node8677) = (Data.node8647, Data.node8677) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8647_eq_data node8677_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8663 row children.1 children.2) row8663_eq_data branches
set_option maxRecDepth 4096 in
theorem row8629_eq_data : row8629 = Data.row8629 := rfl
def node8629 : Graph := .node 8629 row8629 node8609 node8663
set_option maxRecDepth 4096 in
theorem node8629_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8629 :=
  Budget.node_covers _ _ 8629 row8629 node8609 node8663
    node8609_covers node8663_covers (by decide) (by decide) row8629_covers
theorem node8629_eq_data : node8629 = Data.node8629 := by
  have branches : (node8609, node8663) = (Data.node8609, Data.node8663) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8609_eq_data node8663_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8629 row children.1 children.2) row8629_eq_data branches
set_option maxRecDepth 4096 in
theorem row8689_eq_data : row8689 = Data.row8689 := rfl
def node8689 : Graph := .node 8689 row8689 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8689 :=
  Budget.node_covers _ _ 8689 row8689 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8689_covers
theorem node8689_eq_data : node8689 = Data.node8689 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8689 row children.1 children.2) row8689_eq_data branches
set_option maxRecDepth 4096 in
theorem row8693_eq_data : row8693 = Data.row8693 := rfl
def node8693 : Graph := .node 8693 row8693 node8689 Graph.empty
set_option maxRecDepth 4096 in
theorem node8693_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8693 :=
  Budget.node_covers _ _ 8693 row8693 node8689 Graph.empty
    node8689_covers empty_covers (by decide) (by decide) row8693_covers
theorem node8693_eq_data : node8693 = Data.node8693 := by
  have branches : (node8689, Graph.empty) = (Data.node8689, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8689_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8693 row children.1 children.2) row8693_eq_data branches
set_option maxRecDepth 4096 in
theorem row8707_eq_data : row8707 = Data.row8707 := rfl
def node8707 : Graph := .node 8707 row8707 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8707_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8707 :=
  Budget.node_covers _ _ 8707 row8707 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8707_covers
theorem node8707_eq_data : node8707 = Data.node8707 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8707 row children.1 children.2) row8707_eq_data branches
set_option maxRecDepth 4096 in
theorem row8713_eq_data : row8713 = Data.row8713 := rfl
def node8713 : Graph := .node 8713 row8713 node8707 Graph.empty
set_option maxRecDepth 4096 in
theorem node8713_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8713 :=
  Budget.node_covers _ _ 8713 row8713 node8707 Graph.empty
    node8707_covers empty_covers (by decide) (by decide) row8713_covers
theorem node8713_eq_data : node8713 = Data.node8713 := by
  have branches : (node8707, Graph.empty) = (Data.node8707, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8707_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8713 row children.1 children.2) row8713_eq_data branches
set_option maxRecDepth 4096 in
theorem row8699_eq_data : row8699 = Data.row8699 := rfl
def node8699 : Graph := .node 8699 row8699 node8693 node8713
set_option maxRecDepth 4096 in
theorem node8699_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8699 :=
  Budget.node_covers _ _ 8699 row8699 node8693 node8713
    node8693_covers node8713_covers (by decide) (by decide) row8699_covers
theorem node8699_eq_data : node8699 = Data.node8699 := by
  have branches : (node8693, node8713) = (Data.node8693, Data.node8713) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8693_eq_data node8713_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8699 row children.1 children.2) row8699_eq_data branches
set_option maxRecDepth 4096 in
theorem row8731_eq_data : row8731 = Data.row8731 := rfl
def node8731 : Graph := .node 8731 row8731 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8731_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8731 :=
  Budget.node_covers _ _ 8731 row8731 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8731_covers
theorem node8731_eq_data : node8731 = Data.node8731 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8731 row children.1 children.2) row8731_eq_data branches
set_option maxRecDepth 4096 in
theorem row8737_eq_data : row8737 = Data.row8737 := rfl
def node8737 : Graph := .node 8737 row8737 node8731 Graph.empty
set_option maxRecDepth 4096 in
theorem node8737_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8737 :=
  Budget.node_covers _ _ 8737 row8737 node8731 Graph.empty
    node8731_covers empty_covers (by decide) (by decide) row8737_covers
theorem node8737_eq_data : node8737 = Data.node8737 := by
  have branches : (node8731, Graph.empty) = (Data.node8731, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8731_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8737 row children.1 children.2) row8737_eq_data branches
set_option maxRecDepth 4096 in
theorem row8747_eq_data : row8747 = Data.row8747 := rfl
def node8747 : Graph := .node 8747 row8747 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8747_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8747 :=
  Budget.node_covers _ _ 8747 row8747 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8747_covers
theorem node8747_eq_data : node8747 = Data.node8747 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8747 row children.1 children.2) row8747_eq_data branches
set_option maxRecDepth 4096 in
theorem row8753_eq_data : row8753 = Data.row8753 := rfl
def node8753 : Graph := .node 8753 row8753 node8747 Graph.empty
set_option maxRecDepth 4096 in
theorem node8753_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8753 :=
  Budget.node_covers _ _ 8753 row8753 node8747 Graph.empty
    node8747_covers empty_covers (by decide) (by decide) row8753_covers
theorem node8753_eq_data : node8753 = Data.node8753 := by
  have branches : (node8747, Graph.empty) = (Data.node8747, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8747_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8753 row children.1 children.2) row8753_eq_data branches
set_option maxRecDepth 4096 in
theorem row8741_eq_data : row8741 = Data.row8741 := rfl
def node8741 : Graph := .node 8741 row8741 node8737 node8753
set_option maxRecDepth 4096 in
theorem node8741_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8741 :=
  Budget.node_covers _ _ 8741 row8741 node8737 node8753
    node8737_covers node8753_covers (by decide) (by decide) row8741_covers
theorem node8741_eq_data : node8741 = Data.node8741 := by
  have branches : (node8737, node8753) = (Data.node8737, Data.node8753) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8737_eq_data node8753_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8741 row children.1 children.2) row8741_eq_data branches
set_option maxRecDepth 4096 in
theorem row8719_eq_data : row8719 = Data.row8719 := rfl
def node8719 : Graph := .node 8719 row8719 node8699 node8741
set_option maxRecDepth 4096 in
theorem node8719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8719 :=
  Budget.node_covers _ _ 8719 row8719 node8699 node8741
    node8699_covers node8741_covers (by decide) (by decide) row8719_covers
theorem node8719_eq_data : node8719 = Data.node8719 := by
  have branches : (node8699, node8741) = (Data.node8699, Data.node8741) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8699_eq_data node8741_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8719 row children.1 children.2) row8719_eq_data branches
set_option maxRecDepth 4096 in
theorem row8681_eq_data : row8681 = Data.row8681 := rfl
def node8681 : Graph := .node 8681 row8681 node8629 node8719
set_option maxRecDepth 4096 in
theorem node8681_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8681 :=
  Budget.node_covers _ _ 8681 row8681 node8629 node8719
    node8629_covers node8719_covers (by decide) (by decide) row8681_covers
theorem node8681_eq_data : node8681 = Data.node8681 := by
  have branches : (node8629, node8719) = (Data.node8629, Data.node8719) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8629_eq_data node8719_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8681 row children.1 children.2) row8681_eq_data branches
set_option maxRecDepth 4096 in
theorem row8779_eq_data : row8779 = Data.row8779 := rfl
def node8779 : Graph := .node 8779 row8779 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node8779_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8779 :=
  Budget.node_covers _ _ 8779 row8779 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row8779_covers
theorem node8779_eq_data : node8779 = Data.node8779 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8779 row children.1 children.2) row8779_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
