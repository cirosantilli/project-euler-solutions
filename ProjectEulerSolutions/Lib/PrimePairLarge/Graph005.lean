import ProjectEulerSolutions.Lib.PrimePairLarge.Graph004
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row2203_eq_data : row2203 = Data.row2203 := rfl
def node2203 : Graph := .node 2203 row2203 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2203_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2203 :=
  Budget.node_covers _ _ 2203 row2203 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2203_covers
theorem node2203_eq_data : node2203 = Data.node2203 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2203 row children.1 children.2) row2203_eq_data branches
set_option maxRecDepth 4096 in
theorem row2207_eq_data : row2207 = Data.row2207 := rfl
def node2207 : Graph := .node 2207 row2207 node2203 Graph.empty
set_option maxRecDepth 4096 in
theorem node2207_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2207 :=
  Budget.node_covers _ _ 2207 row2207 node2203 Graph.empty
    node2203_covers empty_covers (by decide) (by decide) row2207_covers
theorem node2207_eq_data : node2207 = Data.node2207 := by
  have branches : (node2203, Graph.empty) = (Data.node2203, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2203_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2207 row children.1 children.2) row2207_eq_data branches
set_option maxRecDepth 4096 in
theorem row2179_eq_data : row2179 = Data.row2179 := rfl
def node2179 : Graph := .node 2179 row2179 node2161 node2207
set_option maxRecDepth 4096 in
theorem node2179_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2179 :=
  Budget.node_covers _ _ 2179 row2179 node2161 node2207
    node2161_covers node2207_covers (by decide) (by decide) row2179_covers
theorem node2179_eq_data : node2179 = Data.node2179 := by
  have branches : (node2161, node2207) = (Data.node2161, Data.node2207) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2161_eq_data node2207_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2179 row children.1 children.2) row2179_eq_data branches
set_option maxRecDepth 4096 in
theorem row2143_eq_data : row2143 = Data.row2143 := rfl
def node2143 : Graph := .node 2143 row2143 node2131 node2179
set_option maxRecDepth 4096 in
theorem node2143_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2143 :=
  Budget.node_covers _ _ 2143 row2143 node2131 node2179
    node2131_covers node2179_covers (by decide) (by decide) row2143_covers
theorem node2143_eq_data : node2143 = Data.node2143 := by
  have branches : (node2131, node2179) = (Data.node2131, Data.node2179) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2131_eq_data node2179_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2143 row children.1 children.2) row2143_eq_data branches
set_option maxRecDepth 4096 in
theorem row2221_eq_data : row2221 = Data.row2221 := rfl
def node2221 : Graph := .node 2221 row2221 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2221_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2221 :=
  Budget.node_covers _ _ 2221 row2221 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2221_covers
theorem node2221_eq_data : node2221 = Data.node2221 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2221 row children.1 children.2) row2221_eq_data branches
set_option maxRecDepth 4096 in
theorem row2237_eq_data : row2237 = Data.row2237 := rfl
def node2237 : Graph := .node 2237 row2237 node2221 Graph.empty
set_option maxRecDepth 4096 in
theorem node2237_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2237 :=
  Budget.node_covers _ _ 2237 row2237 node2221 Graph.empty
    node2221_covers empty_covers (by decide) (by decide) row2237_covers
theorem node2237_eq_data : node2237 = Data.node2237 := by
  have branches : (node2221, Graph.empty) = (Data.node2221, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2221_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2237 row children.1 children.2) row2237_eq_data branches
set_option maxRecDepth 4096 in
theorem row2243_eq_data : row2243 = Data.row2243 := rfl
def node2243 : Graph := .node 2243 row2243 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2243_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2243 :=
  Budget.node_covers _ _ 2243 row2243 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2243_covers
theorem node2243_eq_data : node2243 = Data.node2243 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2243 row children.1 children.2) row2243_eq_data branches
set_option maxRecDepth 4096 in
theorem row2251_eq_data : row2251 = Data.row2251 := rfl
def node2251 : Graph := .node 2251 row2251 node2243 Graph.empty
set_option maxRecDepth 4096 in
theorem node2251_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2251 :=
  Budget.node_covers _ _ 2251 row2251 node2243 Graph.empty
    node2243_covers empty_covers (by decide) (by decide) row2251_covers
theorem node2251_eq_data : node2251 = Data.node2251 := by
  have branches : (node2243, Graph.empty) = (Data.node2243, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2243_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2251 row children.1 children.2) row2251_eq_data branches
set_option maxRecDepth 4096 in
theorem row2239_eq_data : row2239 = Data.row2239 := rfl
def node2239 : Graph := .node 2239 row2239 node2237 node2251
set_option maxRecDepth 4096 in
theorem node2239_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2239 :=
  Budget.node_covers _ _ 2239 row2239 node2237 node2251
    node2237_covers node2251_covers (by decide) (by decide) row2239_covers
theorem node2239_eq_data : node2239 = Data.node2239 := by
  have branches : (node2237, node2251) = (Data.node2237, Data.node2251) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2237_eq_data node2251_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2239 row children.1 children.2) row2239_eq_data branches
set_option maxRecDepth 4096 in
theorem row2269_eq_data : row2269 = Data.row2269 := rfl
def node2269 : Graph := .node 2269 row2269 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2269_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2269 :=
  Budget.node_covers _ _ 2269 row2269 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2269_covers
theorem node2269_eq_data : node2269 = Data.node2269 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2269 row children.1 children.2) row2269_eq_data branches
set_option maxRecDepth 4096 in
theorem row2273_eq_data : row2273 = Data.row2273 := rfl
def node2273 : Graph := .node 2273 row2273 node2269 Graph.empty
set_option maxRecDepth 4096 in
theorem node2273_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2273 :=
  Budget.node_covers _ _ 2273 row2273 node2269 Graph.empty
    node2269_covers empty_covers (by decide) (by decide) row2273_covers
theorem node2273_eq_data : node2273 = Data.node2273 := by
  have branches : (node2269, Graph.empty) = (Data.node2269, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2269_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2273 row children.1 children.2) row2273_eq_data branches
set_option maxRecDepth 4096 in
theorem row2287_eq_data : row2287 = Data.row2287 := rfl
def node2287 : Graph := .node 2287 row2287 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2287_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2287 :=
  Budget.node_covers _ _ 2287 row2287 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2287_covers
theorem node2287_eq_data : node2287 = Data.node2287 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2287 row children.1 children.2) row2287_eq_data branches
set_option maxRecDepth 4096 in
theorem row2293_eq_data : row2293 = Data.row2293 := rfl
def node2293 : Graph := .node 2293 row2293 node2287 Graph.empty
set_option maxRecDepth 4096 in
theorem node2293_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2293 :=
  Budget.node_covers _ _ 2293 row2293 node2287 Graph.empty
    node2287_covers empty_covers (by decide) (by decide) row2293_covers
theorem node2293_eq_data : node2293 = Data.node2293 := by
  have branches : (node2287, Graph.empty) = (Data.node2287, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2287_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2293 row children.1 children.2) row2293_eq_data branches
set_option maxRecDepth 4096 in
theorem row2281_eq_data : row2281 = Data.row2281 := rfl
def node2281 : Graph := .node 2281 row2281 node2273 node2293
set_option maxRecDepth 4096 in
theorem node2281_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2281 :=
  Budget.node_covers _ _ 2281 row2281 node2273 node2293
    node2273_covers node2293_covers (by decide) (by decide) row2281_covers
theorem node2281_eq_data : node2281 = Data.node2281 := by
  have branches : (node2273, node2293) = (Data.node2273, Data.node2293) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2273_eq_data node2293_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2281 row children.1 children.2) row2281_eq_data branches
set_option maxRecDepth 4096 in
theorem row2267_eq_data : row2267 = Data.row2267 := rfl
def node2267 : Graph := .node 2267 row2267 node2239 node2281
set_option maxRecDepth 4096 in
theorem node2267_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2267 :=
  Budget.node_covers _ _ 2267 row2267 node2239 node2281
    node2239_covers node2281_covers (by decide) (by decide) row2267_covers
theorem node2267_eq_data : node2267 = Data.node2267 := by
  have branches : (node2239, node2281) = (Data.node2239, Data.node2281) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2239_eq_data node2281_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2267 row children.1 children.2) row2267_eq_data branches
set_option maxRecDepth 4096 in
theorem row2213_eq_data : row2213 = Data.row2213 := rfl
def node2213 : Graph := .node 2213 row2213 node2143 node2267
set_option maxRecDepth 4096 in
theorem node2213_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2213 :=
  Budget.node_covers _ _ 2213 row2213 node2143 node2267
    node2143_covers node2267_covers (by decide) (by decide) row2213_covers
theorem node2213_eq_data : node2213 = Data.node2213 := by
  have branches : (node2143, node2267) = (Data.node2143, Data.node2267) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2143_eq_data node2267_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2213 row children.1 children.2) row2213_eq_data branches
set_option maxRecDepth 4096 in
theorem row2111_eq_data : row2111 = Data.row2111 := rfl
def node2111 : Graph := .node 2111 row2111 node2017 node2213
set_option maxRecDepth 4096 in
theorem node2111_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2111 :=
  Budget.node_covers _ _ 2111 row2111 node2017 node2213
    node2017_covers node2213_covers (by decide) (by decide) row2111_covers
theorem node2111_eq_data : node2111 = Data.node2111 := by
  have branches : (node2017, node2213) = (Data.node2017, Data.node2213) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2017_eq_data node2213_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2111 row children.1 children.2) row2111_eq_data branches
set_option maxRecDepth 4096 in
theorem row2309_eq_data : row2309 = Data.row2309 := rfl
def node2309 : Graph := .node 2309 row2309 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2309_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2309 :=
  Budget.node_covers _ _ 2309 row2309 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2309_covers
theorem node2309_eq_data : node2309 = Data.node2309 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2309 row children.1 children.2) row2309_eq_data branches
set_option maxRecDepth 4096 in
theorem row2311_eq_data : row2311 = Data.row2311 := rfl
def node2311 : Graph := .node 2311 row2311 node2309 Graph.empty
set_option maxRecDepth 4096 in
theorem node2311_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2311 :=
  Budget.node_covers _ _ 2311 row2311 node2309 Graph.empty
    node2309_covers empty_covers (by decide) (by decide) row2311_covers
theorem node2311_eq_data : node2311 = Data.node2311 := by
  have branches : (node2309, Graph.empty) = (Data.node2309, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2309_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2311 row children.1 children.2) row2311_eq_data branches
set_option maxRecDepth 4096 in
theorem row2339_eq_data : row2339 = Data.row2339 := rfl
def node2339 : Graph := .node 2339 row2339 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2339_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2339 :=
  Budget.node_covers _ _ 2339 row2339 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2339_covers
theorem node2339_eq_data : node2339 = Data.node2339 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2339 row children.1 children.2) row2339_eq_data branches
set_option maxRecDepth 4096 in
theorem row2341_eq_data : row2341 = Data.row2341 := rfl
def node2341 : Graph := .node 2341 row2341 node2339 Graph.empty
set_option maxRecDepth 4096 in
theorem node2341_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2341 :=
  Budget.node_covers _ _ 2341 row2341 node2339 Graph.empty
    node2339_covers empty_covers (by decide) (by decide) row2341_covers
theorem node2341_eq_data : node2341 = Data.node2341 := by
  have branches : (node2339, Graph.empty) = (Data.node2339, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2339_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2341 row children.1 children.2) row2341_eq_data branches
set_option maxRecDepth 4096 in
theorem row2333_eq_data : row2333 = Data.row2333 := rfl
def node2333 : Graph := .node 2333 row2333 node2311 node2341
set_option maxRecDepth 4096 in
theorem node2333_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2333 :=
  Budget.node_covers _ _ 2333 row2333 node2311 node2341
    node2311_covers node2341_covers (by decide) (by decide) row2333_covers
theorem node2333_eq_data : node2333 = Data.node2333 := by
  have branches : (node2311, node2341) = (Data.node2311, Data.node2341) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2311_eq_data node2341_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2333 row children.1 children.2) row2333_eq_data branches
set_option maxRecDepth 4096 in
theorem row2351_eq_data : row2351 = Data.row2351 := rfl
def node2351 : Graph := .node 2351 row2351 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2351_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2351 :=
  Budget.node_covers _ _ 2351 row2351 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2351_covers
theorem node2351_eq_data : node2351 = Data.node2351 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2351 row children.1 children.2) row2351_eq_data branches
set_option maxRecDepth 4096 in
theorem row2357_eq_data : row2357 = Data.row2357 := rfl
def node2357 : Graph := .node 2357 row2357 node2351 Graph.empty
set_option maxRecDepth 4096 in
theorem node2357_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2357 :=
  Budget.node_covers _ _ 2357 row2357 node2351 Graph.empty
    node2351_covers empty_covers (by decide) (by decide) row2357_covers
theorem node2357_eq_data : node2357 = Data.node2357 := by
  have branches : (node2351, Graph.empty) = (Data.node2351, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2351_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2357 row children.1 children.2) row2357_eq_data branches
set_option maxRecDepth 4096 in
theorem row2377_eq_data : row2377 = Data.row2377 := rfl
def node2377 : Graph := .node 2377 row2377 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2377_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2377 :=
  Budget.node_covers _ _ 2377 row2377 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2377_covers
theorem node2377_eq_data : node2377 = Data.node2377 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2377 row children.1 children.2) row2377_eq_data branches
set_option maxRecDepth 4096 in
theorem row2381_eq_data : row2381 = Data.row2381 := rfl
def node2381 : Graph := .node 2381 row2381 node2377 Graph.empty
set_option maxRecDepth 4096 in
theorem node2381_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2381 :=
  Budget.node_covers _ _ 2381 row2381 node2377 Graph.empty
    node2377_covers empty_covers (by decide) (by decide) row2381_covers
theorem node2381_eq_data : node2381 = Data.node2381 := by
  have branches : (node2377, Graph.empty) = (Data.node2377, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2377_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2381 row children.1 children.2) row2381_eq_data branches
set_option maxRecDepth 4096 in
theorem row2371_eq_data : row2371 = Data.row2371 := rfl
def node2371 : Graph := .node 2371 row2371 node2357 node2381
set_option maxRecDepth 4096 in
theorem node2371_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2371 :=
  Budget.node_covers _ _ 2371 row2371 node2357 node2381
    node2357_covers node2381_covers (by decide) (by decide) row2371_covers
theorem node2371_eq_data : node2371 = Data.node2371 := by
  have branches : (node2357, node2381) = (Data.node2357, Data.node2381) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2357_eq_data node2381_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2371 row children.1 children.2) row2371_eq_data branches
set_option maxRecDepth 4096 in
theorem row2347_eq_data : row2347 = Data.row2347 := rfl
def node2347 : Graph := .node 2347 row2347 node2333 node2371
set_option maxRecDepth 4096 in
theorem node2347_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2347 :=
  Budget.node_covers _ _ 2347 row2347 node2333 node2371
    node2333_covers node2371_covers (by decide) (by decide) row2347_covers
theorem node2347_eq_data : node2347 = Data.node2347 := by
  have branches : (node2333, node2371) = (Data.node2333, Data.node2371) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2333_eq_data node2371_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2347 row children.1 children.2) row2347_eq_data branches
set_option maxRecDepth 4096 in
theorem row2389_eq_data : row2389 = Data.row2389 := rfl
def node2389 : Graph := .node 2389 row2389 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2389_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2389 :=
  Budget.node_covers _ _ 2389 row2389 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2389_covers
theorem node2389_eq_data : node2389 = Data.node2389 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2389 row children.1 children.2) row2389_eq_data branches
set_option maxRecDepth 4096 in
theorem row2393_eq_data : row2393 = Data.row2393 := rfl
def node2393 : Graph := .node 2393 row2393 node2389 Graph.empty
set_option maxRecDepth 4096 in
theorem node2393_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2393 :=
  Budget.node_covers _ _ 2393 row2393 node2389 Graph.empty
    node2389_covers empty_covers (by decide) (by decide) row2393_covers
theorem node2393_eq_data : node2393 = Data.node2393 := by
  have branches : (node2389, Graph.empty) = (Data.node2389, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2389_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2393 row children.1 children.2) row2393_eq_data branches
set_option maxRecDepth 4096 in
theorem row2411_eq_data : row2411 = Data.row2411 := rfl
def node2411 : Graph := .node 2411 row2411 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2411_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2411 :=
  Budget.node_covers _ _ 2411 row2411 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2411_covers
theorem node2411_eq_data : node2411 = Data.node2411 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2411 row children.1 children.2) row2411_eq_data branches
set_option maxRecDepth 4096 in
theorem row2417_eq_data : row2417 = Data.row2417 := rfl
def node2417 : Graph := .node 2417 row2417 node2411 Graph.empty
set_option maxRecDepth 4096 in
theorem node2417_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2417 :=
  Budget.node_covers _ _ 2417 row2417 node2411 Graph.empty
    node2411_covers empty_covers (by decide) (by decide) row2417_covers
theorem node2417_eq_data : node2417 = Data.node2417 := by
  have branches : (node2411, Graph.empty) = (Data.node2411, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2411_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2417 row children.1 children.2) row2417_eq_data branches
set_option maxRecDepth 4096 in
theorem row2399_eq_data : row2399 = Data.row2399 := rfl
def node2399 : Graph := .node 2399 row2399 node2393 node2417
set_option maxRecDepth 4096 in
theorem node2399_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2399 :=
  Budget.node_covers _ _ 2399 row2399 node2393 node2417
    node2393_covers node2417_covers (by decide) (by decide) row2399_covers
theorem node2399_eq_data : node2399 = Data.node2399 := by
  have branches : (node2393, node2417) = (Data.node2393, Data.node2417) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2393_eq_data node2417_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2399 row children.1 children.2) row2399_eq_data branches
set_option maxRecDepth 4096 in
theorem row2437_eq_data : row2437 = Data.row2437 := rfl
def node2437 : Graph := .node 2437 row2437 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2437_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2437 :=
  Budget.node_covers _ _ 2437 row2437 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2437_covers
theorem node2437_eq_data : node2437 = Data.node2437 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2437 row children.1 children.2) row2437_eq_data branches
set_option maxRecDepth 4096 in
theorem row2441_eq_data : row2441 = Data.row2441 := rfl
def node2441 : Graph := .node 2441 row2441 node2437 Graph.empty
set_option maxRecDepth 4096 in
theorem node2441_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2441 :=
  Budget.node_covers _ _ 2441 row2441 node2437 Graph.empty
    node2437_covers empty_covers (by decide) (by decide) row2441_covers
theorem node2441_eq_data : node2441 = Data.node2441 := by
  have branches : (node2437, Graph.empty) = (Data.node2437, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2437_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2441 row children.1 children.2) row2441_eq_data branches
set_option maxRecDepth 4096 in
theorem row2459_eq_data : row2459 = Data.row2459 := rfl
def node2459 : Graph := .node 2459 row2459 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2459_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2459 :=
  Budget.node_covers _ _ 2459 row2459 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2459_covers
theorem node2459_eq_data : node2459 = Data.node2459 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2459 row children.1 children.2) row2459_eq_data branches
set_option maxRecDepth 4096 in
theorem row2467_eq_data : row2467 = Data.row2467 := rfl
def node2467 : Graph := .node 2467 row2467 node2459 Graph.empty
set_option maxRecDepth 4096 in
theorem node2467_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2467 :=
  Budget.node_covers _ _ 2467 row2467 node2459 Graph.empty
    node2459_covers empty_covers (by decide) (by decide) row2467_covers
theorem node2467_eq_data : node2467 = Data.node2467 := by
  have branches : (node2459, Graph.empty) = (Data.node2459, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2459_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2467 row children.1 children.2) row2467_eq_data branches
set_option maxRecDepth 4096 in
theorem row2447_eq_data : row2447 = Data.row2447 := rfl
def node2447 : Graph := .node 2447 row2447 node2441 node2467
set_option maxRecDepth 4096 in
theorem node2447_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2447 :=
  Budget.node_covers _ _ 2447 row2447 node2441 node2467
    node2441_covers node2467_covers (by decide) (by decide) row2447_covers
theorem node2447_eq_data : node2447 = Data.node2447 := by
  have branches : (node2441, node2467) = (Data.node2441, Data.node2467) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2441_eq_data node2467_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2447 row children.1 children.2) row2447_eq_data branches
set_option maxRecDepth 4096 in
theorem row2423_eq_data : row2423 = Data.row2423 := rfl
def node2423 : Graph := .node 2423 row2423 node2399 node2447
set_option maxRecDepth 4096 in
theorem node2423_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2423 :=
  Budget.node_covers _ _ 2423 row2423 node2399 node2447
    node2399_covers node2447_covers (by decide) (by decide) row2423_covers
theorem node2423_eq_data : node2423 = Data.node2423 := by
  have branches : (node2399, node2447) = (Data.node2399, Data.node2447) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2399_eq_data node2447_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2423 row children.1 children.2) row2423_eq_data branches
set_option maxRecDepth 4096 in
theorem row2383_eq_data : row2383 = Data.row2383 := rfl
def node2383 : Graph := .node 2383 row2383 node2347 node2423
set_option maxRecDepth 4096 in
theorem node2383_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2383 :=
  Budget.node_covers _ _ 2383 row2383 node2347 node2423
    node2347_covers node2423_covers (by decide) (by decide) row2383_covers
theorem node2383_eq_data : node2383 = Data.node2383 := by
  have branches : (node2347, node2423) = (Data.node2347, Data.node2423) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2347_eq_data node2423_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2383 row children.1 children.2) row2383_eq_data branches
set_option maxRecDepth 4096 in
theorem row2477_eq_data : row2477 = Data.row2477 := rfl
def node2477 : Graph := .node 2477 row2477 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2477_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2477 :=
  Budget.node_covers _ _ 2477 row2477 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2477_covers
theorem node2477_eq_data : node2477 = Data.node2477 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2477 row children.1 children.2) row2477_eq_data branches
set_option maxRecDepth 4096 in
theorem row2503_eq_data : row2503 = Data.row2503 := rfl
def node2503 : Graph := .node 2503 row2503 node2477 Graph.empty
set_option maxRecDepth 4096 in
theorem node2503_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2503 :=
  Budget.node_covers _ _ 2503 row2503 node2477 Graph.empty
    node2477_covers empty_covers (by decide) (by decide) row2503_covers
theorem node2503_eq_data : node2503 = Data.node2503 := by
  have branches : (node2477, Graph.empty) = (Data.node2477, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2477_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2503 row children.1 children.2) row2503_eq_data branches
set_option maxRecDepth 4096 in
theorem row2531_eq_data : row2531 = Data.row2531 := rfl
def node2531 : Graph := .node 2531 row2531 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2531_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2531 :=
  Budget.node_covers _ _ 2531 row2531 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2531_covers
theorem node2531_eq_data : node2531 = Data.node2531 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2531 row children.1 children.2) row2531_eq_data branches
set_option maxRecDepth 4096 in
theorem row2539_eq_data : row2539 = Data.row2539 := rfl
def node2539 : Graph := .node 2539 row2539 node2531 Graph.empty
set_option maxRecDepth 4096 in
theorem node2539_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2539 :=
  Budget.node_covers _ _ 2539 row2539 node2531 Graph.empty
    node2531_covers empty_covers (by decide) (by decide) row2539_covers
theorem node2539_eq_data : node2539 = Data.node2539 := by
  have branches : (node2531, Graph.empty) = (Data.node2531, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2531_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2539 row children.1 children.2) row2539_eq_data branches
set_option maxRecDepth 4096 in
theorem row2521_eq_data : row2521 = Data.row2521 := rfl
def node2521 : Graph := .node 2521 row2521 node2503 node2539
set_option maxRecDepth 4096 in
theorem node2521_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2521 :=
  Budget.node_covers _ _ 2521 row2521 node2503 node2539
    node2503_covers node2539_covers (by decide) (by decide) row2521_covers
theorem node2521_eq_data : node2521 = Data.node2521 := by
  have branches : (node2503, node2539) = (Data.node2503, Data.node2539) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2503_eq_data node2539_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2521 row children.1 children.2) row2521_eq_data branches
set_option maxRecDepth 4096 in
theorem row2549_eq_data : row2549 = Data.row2549 := rfl
def node2549 : Graph := .node 2549 row2549 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2549_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2549 :=
  Budget.node_covers _ _ 2549 row2549 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2549_covers
theorem node2549_eq_data : node2549 = Data.node2549 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2549 row children.1 children.2) row2549_eq_data branches
set_option maxRecDepth 4096 in
theorem row2551_eq_data : row2551 = Data.row2551 := rfl
def node2551 : Graph := .node 2551 row2551 node2549 Graph.empty
set_option maxRecDepth 4096 in
theorem node2551_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2551 :=
  Budget.node_covers _ _ 2551 row2551 node2549 Graph.empty
    node2549_covers empty_covers (by decide) (by decide) row2551_covers
theorem node2551_eq_data : node2551 = Data.node2551 := by
  have branches : (node2549, Graph.empty) = (Data.node2549, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2549_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2551 row children.1 children.2) row2551_eq_data branches
set_option maxRecDepth 4096 in
theorem row2579_eq_data : row2579 = Data.row2579 := rfl
def node2579 : Graph := .node 2579 row2579 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2579_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2579 :=
  Budget.node_covers _ _ 2579 row2579 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2579_covers
theorem node2579_eq_data : node2579 = Data.node2579 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2579 row children.1 children.2) row2579_eq_data branches
set_option maxRecDepth 4096 in
theorem row2591_eq_data : row2591 = Data.row2591 := rfl
def node2591 : Graph := .node 2591 row2591 node2579 Graph.empty
set_option maxRecDepth 4096 in
theorem node2591_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2591 :=
  Budget.node_covers _ _ 2591 row2591 node2579 Graph.empty
    node2579_covers empty_covers (by decide) (by decide) row2591_covers
theorem node2591_eq_data : node2591 = Data.node2591 := by
  have branches : (node2579, Graph.empty) = (Data.node2579, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2579_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2591 row children.1 children.2) row2591_eq_data branches
set_option maxRecDepth 4096 in
theorem row2557_eq_data : row2557 = Data.row2557 := rfl
def node2557 : Graph := .node 2557 row2557 node2551 node2591
set_option maxRecDepth 4096 in
theorem node2557_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2557 :=
  Budget.node_covers _ _ 2557 row2557 node2551 node2591
    node2551_covers node2591_covers (by decide) (by decide) row2557_covers
theorem node2557_eq_data : node2557 = Data.node2557 := by
  have branches : (node2551, node2591) = (Data.node2551, Data.node2591) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2551_eq_data node2591_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2557 row children.1 children.2) row2557_eq_data branches
set_option maxRecDepth 4096 in
theorem row2543_eq_data : row2543 = Data.row2543 := rfl
def node2543 : Graph := .node 2543 row2543 node2521 node2557
set_option maxRecDepth 4096 in
theorem node2543_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2543 :=
  Budget.node_covers _ _ 2543 row2543 node2521 node2557
    node2521_covers node2557_covers (by decide) (by decide) row2543_covers
theorem node2543_eq_data : node2543 = Data.node2543 := by
  have branches : (node2521, node2557) = (Data.node2521, Data.node2557) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2521_eq_data node2557_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2543 row children.1 children.2) row2543_eq_data branches
set_option maxRecDepth 4096 in
theorem row2609_eq_data : row2609 = Data.row2609 := rfl
def node2609 : Graph := .node 2609 row2609 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2609_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2609 :=
  Budget.node_covers _ _ 2609 row2609 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2609_covers
theorem node2609_eq_data : node2609 = Data.node2609 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2609 row children.1 children.2) row2609_eq_data branches
set_option maxRecDepth 4096 in
theorem row2617_eq_data : row2617 = Data.row2617 := rfl
def node2617 : Graph := .node 2617 row2617 node2609 Graph.empty
set_option maxRecDepth 4096 in
theorem node2617_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2617 :=
  Budget.node_covers _ _ 2617 row2617 node2609 Graph.empty
    node2609_covers empty_covers (by decide) (by decide) row2617_covers
theorem node2617_eq_data : node2617 = Data.node2617 := by
  have branches : (node2609, Graph.empty) = (Data.node2609, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2609_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2617 row children.1 children.2) row2617_eq_data branches
set_option maxRecDepth 4096 in
theorem row2633_eq_data : row2633 = Data.row2633 := rfl
def node2633 : Graph := .node 2633 row2633 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2633_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2633 :=
  Budget.node_covers _ _ 2633 row2633 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2633_covers
theorem node2633_eq_data : node2633 = Data.node2633 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2633 row children.1 children.2) row2633_eq_data branches
set_option maxRecDepth 4096 in
theorem row2647_eq_data : row2647 = Data.row2647 := rfl
def node2647 : Graph := .node 2647 row2647 node2633 Graph.empty
set_option maxRecDepth 4096 in
theorem node2647_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2647 :=
  Budget.node_covers _ _ 2647 row2647 node2633 Graph.empty
    node2633_covers empty_covers (by decide) (by decide) row2647_covers
theorem node2647_eq_data : node2647 = Data.node2647 := by
  have branches : (node2633, Graph.empty) = (Data.node2633, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2633_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2647 row children.1 children.2) row2647_eq_data branches
set_option maxRecDepth 4096 in
theorem row2621_eq_data : row2621 = Data.row2621 := rfl
def node2621 : Graph := .node 2621 row2621 node2617 node2647
set_option maxRecDepth 4096 in
theorem node2621_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2621 :=
  Budget.node_covers _ _ 2621 row2621 node2617 node2647
    node2617_covers node2647_covers (by decide) (by decide) row2621_covers
theorem node2621_eq_data : node2621 = Data.node2621 := by
  have branches : (node2617, node2647) = (Data.node2617, Data.node2647) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2617_eq_data node2647_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2621 row children.1 children.2) row2621_eq_data branches
set_option maxRecDepth 4096 in
theorem row2659_eq_data : row2659 = Data.row2659 := rfl
def node2659 : Graph := .node 2659 row2659 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2659_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2659 :=
  Budget.node_covers _ _ 2659 row2659 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2659_covers
theorem node2659_eq_data : node2659 = Data.node2659 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2659 row children.1 children.2) row2659_eq_data branches
set_option maxRecDepth 4096 in
theorem row2663_eq_data : row2663 = Data.row2663 := rfl
def node2663 : Graph := .node 2663 row2663 node2659 Graph.empty
set_option maxRecDepth 4096 in
theorem node2663_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2663 :=
  Budget.node_covers _ _ 2663 row2663 node2659 Graph.empty
    node2659_covers empty_covers (by decide) (by decide) row2663_covers
theorem node2663_eq_data : node2663 = Data.node2663 := by
  have branches : (node2659, Graph.empty) = (Data.node2659, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2659_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2663 row children.1 children.2) row2663_eq_data branches
set_option maxRecDepth 4096 in
theorem row2677_eq_data : row2677 = Data.row2677 := rfl
def node2677 : Graph := .node 2677 row2677 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2677_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2677 :=
  Budget.node_covers _ _ 2677 row2677 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2677_covers
theorem node2677_eq_data : node2677 = Data.node2677 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2677 row children.1 children.2) row2677_eq_data branches
set_option maxRecDepth 4096 in
theorem row2683_eq_data : row2683 = Data.row2683 := rfl
def node2683 : Graph := .node 2683 row2683 node2677 Graph.empty
set_option maxRecDepth 4096 in
theorem node2683_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2683 :=
  Budget.node_covers _ _ 2683 row2683 node2677 Graph.empty
    node2677_covers empty_covers (by decide) (by decide) row2683_covers
theorem node2683_eq_data : node2683 = Data.node2683 := by
  have branches : (node2677, Graph.empty) = (Data.node2677, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2677_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2683 row children.1 children.2) row2683_eq_data branches
set_option maxRecDepth 4096 in
theorem row2671_eq_data : row2671 = Data.row2671 := rfl
def node2671 : Graph := .node 2671 row2671 node2663 node2683
set_option maxRecDepth 4096 in
theorem node2671_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2671 :=
  Budget.node_covers _ _ 2671 row2671 node2663 node2683
    node2663_covers node2683_covers (by decide) (by decide) row2671_covers
theorem node2671_eq_data : node2671 = Data.node2671 := by
  have branches : (node2663, node2683) = (Data.node2663, Data.node2683) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2663_eq_data node2683_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2671 row children.1 children.2) row2671_eq_data branches
set_option maxRecDepth 4096 in
theorem row2657_eq_data : row2657 = Data.row2657 := rfl
def node2657 : Graph := .node 2657 row2657 node2621 node2671
set_option maxRecDepth 4096 in
theorem node2657_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2657 :=
  Budget.node_covers _ _ 2657 row2657 node2621 node2671
    node2621_covers node2671_covers (by decide) (by decide) row2657_covers
theorem node2657_eq_data : node2657 = Data.node2657 := by
  have branches : (node2621, node2671) = (Data.node2621, Data.node2671) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2621_eq_data node2671_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2657 row children.1 children.2) row2657_eq_data branches
set_option maxRecDepth 4096 in
theorem row2593_eq_data : row2593 = Data.row2593 := rfl
def node2593 : Graph := .node 2593 row2593 node2543 node2657
set_option maxRecDepth 4096 in
theorem node2593_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2593 :=
  Budget.node_covers _ _ 2593 row2593 node2543 node2657
    node2543_covers node2657_covers (by decide) (by decide) row2593_covers
theorem node2593_eq_data : node2593 = Data.node2593 := by
  have branches : (node2543, node2657) = (Data.node2543, Data.node2657) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2543_eq_data node2657_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2593 row children.1 children.2) row2593_eq_data branches
set_option maxRecDepth 4096 in
theorem row2473_eq_data : row2473 = Data.row2473 := rfl
def node2473 : Graph := .node 2473 row2473 node2383 node2593
set_option maxRecDepth 4096 in
theorem node2473_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2473 :=
  Budget.node_covers _ _ 2473 row2473 node2383 node2593
    node2383_covers node2593_covers (by decide) (by decide) row2473_covers
theorem node2473_eq_data : node2473 = Data.node2473 := by
  have branches : (node2383, node2593) = (Data.node2383, Data.node2593) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2383_eq_data node2593_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2473 row children.1 children.2) row2473_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
