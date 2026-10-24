import ProjectEulerSolutions.Lib.PrimePairLarge.Graph006
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row3217_eq_data : row3217 = Data.row3217 := rfl
def node3217 : Graph := .node 3217 row3217 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3217_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3217 :=
  Budget.node_covers _ _ 3217 row3217 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3217_covers
theorem node3217_eq_data : node3217 = Data.node3217 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3217 row children.1 children.2) row3217_eq_data branches
set_option maxRecDepth 4096 in
theorem row3221_eq_data : row3221 = Data.row3221 := rfl
def node3221 : Graph := .node 3221 row3221 node3217 Graph.empty
set_option maxRecDepth 4096 in
theorem node3221_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3221 :=
  Budget.node_covers _ _ 3221 row3221 node3217 Graph.empty
    node3217_covers empty_covers (by decide) (by decide) row3221_covers
theorem node3221_eq_data : node3221 = Data.node3221 := by
  have branches : (node3217, Graph.empty) = (Data.node3217, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3217_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3221 row children.1 children.2) row3221_eq_data branches
set_option maxRecDepth 4096 in
theorem row3209_eq_data : row3209 = Data.row3209 := rfl
def node3209 : Graph := .node 3209 row3209 node3203 node3221
set_option maxRecDepth 4096 in
theorem node3209_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3209 :=
  Budget.node_covers _ _ 3209 row3209 node3203 node3221
    node3203_covers node3221_covers (by decide) (by decide) row3209_covers
theorem node3209_eq_data : node3209 = Data.node3209 := by
  have branches : (node3203, node3221) = (Data.node3203, Data.node3221) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3203_eq_data node3221_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3209 row children.1 children.2) row3209_eq_data branches
set_option maxRecDepth 4096 in
theorem row3251_eq_data : row3251 = Data.row3251 := rfl
def node3251 : Graph := .node 3251 row3251 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3251_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3251 :=
  Budget.node_covers _ _ 3251 row3251 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3251_covers
theorem node3251_eq_data : node3251 = Data.node3251 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3251 row children.1 children.2) row3251_eq_data branches
set_option maxRecDepth 4096 in
theorem row3253_eq_data : row3253 = Data.row3253 := rfl
def node3253 : Graph := .node 3253 row3253 node3251 Graph.empty
set_option maxRecDepth 4096 in
theorem node3253_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3253 :=
  Budget.node_covers _ _ 3253 row3253 node3251 Graph.empty
    node3251_covers empty_covers (by decide) (by decide) row3253_covers
theorem node3253_eq_data : node3253 = Data.node3253 := by
  have branches : (node3251, Graph.empty) = (Data.node3251, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3251_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3253 row children.1 children.2) row3253_eq_data branches
set_option maxRecDepth 4096 in
theorem row3259_eq_data : row3259 = Data.row3259 := rfl
def node3259 : Graph := .node 3259 row3259 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3259_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3259 :=
  Budget.node_covers _ _ 3259 row3259 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3259_covers
theorem node3259_eq_data : node3259 = Data.node3259 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3259 row children.1 children.2) row3259_eq_data branches
set_option maxRecDepth 4096 in
theorem row3271_eq_data : row3271 = Data.row3271 := rfl
def node3271 : Graph := .node 3271 row3271 node3259 Graph.empty
set_option maxRecDepth 4096 in
theorem node3271_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3271 :=
  Budget.node_covers _ _ 3271 row3271 node3259 Graph.empty
    node3259_covers empty_covers (by decide) (by decide) row3271_covers
theorem node3271_eq_data : node3271 = Data.node3271 := by
  have branches : (node3259, Graph.empty) = (Data.node3259, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3259_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3271 row children.1 children.2) row3271_eq_data branches
set_option maxRecDepth 4096 in
theorem row3257_eq_data : row3257 = Data.row3257 := rfl
def node3257 : Graph := .node 3257 row3257 node3253 node3271
set_option maxRecDepth 4096 in
theorem node3257_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3257 :=
  Budget.node_covers _ _ 3257 row3257 node3253 node3271
    node3253_covers node3271_covers (by decide) (by decide) row3257_covers
theorem node3257_eq_data : node3257 = Data.node3257 := by
  have branches : (node3253, node3271) = (Data.node3253, Data.node3271) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3253_eq_data node3271_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3257 row children.1 children.2) row3257_eq_data branches
set_option maxRecDepth 4096 in
theorem row3229_eq_data : row3229 = Data.row3229 := rfl
def node3229 : Graph := .node 3229 row3229 node3209 node3257
set_option maxRecDepth 4096 in
theorem node3229_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3229 :=
  Budget.node_covers _ _ 3229 row3229 node3209 node3257
    node3209_covers node3257_covers (by decide) (by decide) row3229_covers
theorem node3229_eq_data : node3229 = Data.node3229 := by
  have branches : (node3209, node3257) = (Data.node3209, Data.node3257) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3209_eq_data node3257_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3229 row children.1 children.2) row3229_eq_data branches
set_option maxRecDepth 4096 in
theorem row3187_eq_data : row3187 = Data.row3187 := rfl
def node3187 : Graph := .node 3187 row3187 node3121 node3229
set_option maxRecDepth 4096 in
theorem node3187_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3187 :=
  Budget.node_covers _ _ 3187 row3187 node3121 node3229
    node3121_covers node3229_covers (by decide) (by decide) row3187_covers
theorem node3187_eq_data : node3187 = Data.node3187 := by
  have branches : (node3121, node3229) = (Data.node3121, Data.node3229) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3121_eq_data node3229_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3187 row children.1 children.2) row3187_eq_data branches
set_option maxRecDepth 4096 in
theorem row3301_eq_data : row3301 = Data.row3301 := rfl
def node3301 : Graph := .node 3301 row3301 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3301_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3301 :=
  Budget.node_covers _ _ 3301 row3301 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3301_covers
theorem node3301_eq_data : node3301 = Data.node3301 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3301 row children.1 children.2) row3301_eq_data branches
set_option maxRecDepth 4096 in
theorem row3307_eq_data : row3307 = Data.row3307 := rfl
def node3307 : Graph := .node 3307 row3307 node3301 Graph.empty
set_option maxRecDepth 4096 in
theorem node3307_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3307 :=
  Budget.node_covers _ _ 3307 row3307 node3301 Graph.empty
    node3301_covers empty_covers (by decide) (by decide) row3307_covers
theorem node3307_eq_data : node3307 = Data.node3307 := by
  have branches : (node3301, Graph.empty) = (Data.node3301, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3301_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3307 row children.1 children.2) row3307_eq_data branches
set_option maxRecDepth 4096 in
theorem row3319_eq_data : row3319 = Data.row3319 := rfl
def node3319 : Graph := .node 3319 row3319 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3319_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3319 :=
  Budget.node_covers _ _ 3319 row3319 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3319_covers
theorem node3319_eq_data : node3319 = Data.node3319 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3319 row children.1 children.2) row3319_eq_data branches
set_option maxRecDepth 4096 in
theorem row3323_eq_data : row3323 = Data.row3323 := rfl
def node3323 : Graph := .node 3323 row3323 node3319 Graph.empty
set_option maxRecDepth 4096 in
theorem node3323_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3323 :=
  Budget.node_covers _ _ 3323 row3323 node3319 Graph.empty
    node3319_covers empty_covers (by decide) (by decide) row3323_covers
theorem node3323_eq_data : node3323 = Data.node3323 := by
  have branches : (node3319, Graph.empty) = (Data.node3319, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3319_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3323 row children.1 children.2) row3323_eq_data branches
set_option maxRecDepth 4096 in
theorem row3313_eq_data : row3313 = Data.row3313 := rfl
def node3313 : Graph := .node 3313 row3313 node3307 node3323
set_option maxRecDepth 4096 in
theorem node3313_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3313 :=
  Budget.node_covers _ _ 3313 row3313 node3307 node3323
    node3307_covers node3323_covers (by decide) (by decide) row3313_covers
theorem node3313_eq_data : node3313 = Data.node3313 := by
  have branches : (node3307, node3323) = (Data.node3307, Data.node3323) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3307_eq_data node3323_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3313 row children.1 children.2) row3313_eq_data branches
set_option maxRecDepth 4096 in
theorem row3331_eq_data : row3331 = Data.row3331 := rfl
def node3331 : Graph := .node 3331 row3331 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3331_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3331 :=
  Budget.node_covers _ _ 3331 row3331 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3331_covers
theorem node3331_eq_data : node3331 = Data.node3331 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3331 row children.1 children.2) row3331_eq_data branches
set_option maxRecDepth 4096 in
theorem row3343_eq_data : row3343 = Data.row3343 := rfl
def node3343 : Graph := .node 3343 row3343 node3331 Graph.empty
set_option maxRecDepth 4096 in
theorem node3343_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3343 :=
  Budget.node_covers _ _ 3343 row3343 node3331 Graph.empty
    node3331_covers empty_covers (by decide) (by decide) row3343_covers
theorem node3343_eq_data : node3343 = Data.node3343 := by
  have branches : (node3331, Graph.empty) = (Data.node3331, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3331_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3343 row children.1 children.2) row3343_eq_data branches
set_option maxRecDepth 4096 in
theorem row3359_eq_data : row3359 = Data.row3359 := rfl
def node3359 : Graph := .node 3359 row3359 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3359_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3359 :=
  Budget.node_covers _ _ 3359 row3359 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3359_covers
theorem node3359_eq_data : node3359 = Data.node3359 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3359 row children.1 children.2) row3359_eq_data branches
set_option maxRecDepth 4096 in
theorem row3361_eq_data : row3361 = Data.row3361 := rfl
def node3361 : Graph := .node 3361 row3361 node3359 Graph.empty
set_option maxRecDepth 4096 in
theorem node3361_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3361 :=
  Budget.node_covers _ _ 3361 row3361 node3359 Graph.empty
    node3359_covers empty_covers (by decide) (by decide) row3361_covers
theorem node3361_eq_data : node3361 = Data.node3361 := by
  have branches : (node3359, Graph.empty) = (Data.node3359, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3359_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3361 row children.1 children.2) row3361_eq_data branches
set_option maxRecDepth 4096 in
theorem row3347_eq_data : row3347 = Data.row3347 := rfl
def node3347 : Graph := .node 3347 row3347 node3343 node3361
set_option maxRecDepth 4096 in
theorem node3347_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3347 :=
  Budget.node_covers _ _ 3347 row3347 node3343 node3361
    node3343_covers node3361_covers (by decide) (by decide) row3347_covers
theorem node3347_eq_data : node3347 = Data.node3347 := by
  have branches : (node3343, node3361) = (Data.node3343, Data.node3361) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3343_eq_data node3361_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3347 row children.1 children.2) row3347_eq_data branches
set_option maxRecDepth 4096 in
theorem row3329_eq_data : row3329 = Data.row3329 := rfl
def node3329 : Graph := .node 3329 row3329 node3313 node3347
set_option maxRecDepth 4096 in
theorem node3329_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3329 :=
  Budget.node_covers _ _ 3329 row3329 node3313 node3347
    node3313_covers node3347_covers (by decide) (by decide) row3329_covers
theorem node3329_eq_data : node3329 = Data.node3329 := by
  have branches : (node3313, node3347) = (Data.node3313, Data.node3347) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3313_eq_data node3347_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3329 row children.1 children.2) row3329_eq_data branches
set_option maxRecDepth 4096 in
theorem row3373_eq_data : row3373 = Data.row3373 := rfl
def node3373 : Graph := .node 3373 row3373 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3373_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3373 :=
  Budget.node_covers _ _ 3373 row3373 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3373_covers
theorem node3373_eq_data : node3373 = Data.node3373 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3373 row children.1 children.2) row3373_eq_data branches
set_option maxRecDepth 4096 in
theorem row3389_eq_data : row3389 = Data.row3389 := rfl
def node3389 : Graph := .node 3389 row3389 node3373 Graph.empty
set_option maxRecDepth 4096 in
theorem node3389_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3389 :=
  Budget.node_covers _ _ 3389 row3389 node3373 Graph.empty
    node3373_covers empty_covers (by decide) (by decide) row3389_covers
theorem node3389_eq_data : node3389 = Data.node3389 := by
  have branches : (node3373, Graph.empty) = (Data.node3373, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3373_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3389 row children.1 children.2) row3389_eq_data branches
set_option maxRecDepth 4096 in
theorem row3407_eq_data : row3407 = Data.row3407 := rfl
def node3407 : Graph := .node 3407 row3407 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3407_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3407 :=
  Budget.node_covers _ _ 3407 row3407 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3407_covers
theorem node3407_eq_data : node3407 = Data.node3407 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3407 row children.1 children.2) row3407_eq_data branches
set_option maxRecDepth 4096 in
theorem row3413_eq_data : row3413 = Data.row3413 := rfl
def node3413 : Graph := .node 3413 row3413 node3407 Graph.empty
set_option maxRecDepth 4096 in
theorem node3413_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3413 :=
  Budget.node_covers _ _ 3413 row3413 node3407 Graph.empty
    node3407_covers empty_covers (by decide) (by decide) row3413_covers
theorem node3413_eq_data : node3413 = Data.node3413 := by
  have branches : (node3407, Graph.empty) = (Data.node3407, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3407_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3413 row children.1 children.2) row3413_eq_data branches
set_option maxRecDepth 4096 in
theorem row3391_eq_data : row3391 = Data.row3391 := rfl
def node3391 : Graph := .node 3391 row3391 node3389 node3413
set_option maxRecDepth 4096 in
theorem node3391_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3391 :=
  Budget.node_covers _ _ 3391 row3391 node3389 node3413
    node3389_covers node3413_covers (by decide) (by decide) row3391_covers
theorem node3391_eq_data : node3391 = Data.node3391 := by
  have branches : (node3389, node3413) = (Data.node3389, Data.node3413) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3389_eq_data node3413_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3391 row children.1 children.2) row3391_eq_data branches
set_option maxRecDepth 4096 in
theorem row3449_eq_data : row3449 = Data.row3449 := rfl
def node3449 : Graph := .node 3449 row3449 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3449_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3449 :=
  Budget.node_covers _ _ 3449 row3449 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3449_covers
theorem node3449_eq_data : node3449 = Data.node3449 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3449 row children.1 children.2) row3449_eq_data branches
set_option maxRecDepth 4096 in
theorem row3457_eq_data : row3457 = Data.row3457 := rfl
def node3457 : Graph := .node 3457 row3457 node3449 Graph.empty
set_option maxRecDepth 4096 in
theorem node3457_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3457 :=
  Budget.node_covers _ _ 3457 row3457 node3449 Graph.empty
    node3449_covers empty_covers (by decide) (by decide) row3457_covers
theorem node3457_eq_data : node3457 = Data.node3457 := by
  have branches : (node3449, Graph.empty) = (Data.node3449, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3449_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3457 row children.1 children.2) row3457_eq_data branches
set_option maxRecDepth 4096 in
theorem row3463_eq_data : row3463 = Data.row3463 := rfl
def node3463 : Graph := .node 3463 row3463 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3463_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3463 :=
  Budget.node_covers _ _ 3463 row3463 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3463_covers
theorem node3463_eq_data : node3463 = Data.node3463 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3463 row children.1 children.2) row3463_eq_data branches
set_option maxRecDepth 4096 in
theorem row3467_eq_data : row3467 = Data.row3467 := rfl
def node3467 : Graph := .node 3467 row3467 node3463 Graph.empty
set_option maxRecDepth 4096 in
theorem node3467_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3467 :=
  Budget.node_covers _ _ 3467 row3467 node3463 Graph.empty
    node3463_covers empty_covers (by decide) (by decide) row3467_covers
theorem node3467_eq_data : node3467 = Data.node3467 := by
  have branches : (node3463, Graph.empty) = (Data.node3463, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3463_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3467 row children.1 children.2) row3467_eq_data branches
set_option maxRecDepth 4096 in
theorem row3461_eq_data : row3461 = Data.row3461 := rfl
def node3461 : Graph := .node 3461 row3461 node3457 node3467
set_option maxRecDepth 4096 in
theorem node3461_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3461 :=
  Budget.node_covers _ _ 3461 row3461 node3457 node3467
    node3457_covers node3467_covers (by decide) (by decide) row3461_covers
theorem node3461_eq_data : node3461 = Data.node3461 := by
  have branches : (node3457, node3467) = (Data.node3457, Data.node3467) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3457_eq_data node3467_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3461 row children.1 children.2) row3461_eq_data branches
set_option maxRecDepth 4096 in
theorem row3433_eq_data : row3433 = Data.row3433 := rfl
def node3433 : Graph := .node 3433 row3433 node3391 node3461
set_option maxRecDepth 4096 in
theorem node3433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3433 :=
  Budget.node_covers _ _ 3433 row3433 node3391 node3461
    node3391_covers node3461_covers (by decide) (by decide) row3433_covers
theorem node3433_eq_data : node3433 = Data.node3433 := by
  have branches : (node3391, node3461) = (Data.node3391, Data.node3461) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3391_eq_data node3461_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3433 row children.1 children.2) row3433_eq_data branches
set_option maxRecDepth 4096 in
theorem row3371_eq_data : row3371 = Data.row3371 := rfl
def node3371 : Graph := .node 3371 row3371 node3329 node3433
set_option maxRecDepth 4096 in
theorem node3371_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3371 :=
  Budget.node_covers _ _ 3371 row3371 node3329 node3433
    node3329_covers node3433_covers (by decide) (by decide) row3371_covers
theorem node3371_eq_data : node3371 = Data.node3371 := by
  have branches : (node3329, node3433) = (Data.node3329, Data.node3433) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3329_eq_data node3433_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3371 row children.1 children.2) row3371_eq_data branches
set_option maxRecDepth 4096 in
theorem row3299_eq_data : row3299 = Data.row3299 := rfl
def node3299 : Graph := .node 3299 row3299 node3187 node3371
set_option maxRecDepth 4096 in
theorem node3299_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3299 :=
  Budget.node_covers _ _ 3299 row3299 node3187 node3371
    node3187_covers node3371_covers (by decide) (by decide) row3299_covers
theorem node3299_eq_data : node3299 = Data.node3299 := by
  have branches : (node3187, node3371) = (Data.node3187, Data.node3371) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3187_eq_data node3371_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3299 row children.1 children.2) row3299_eq_data branches
set_option maxRecDepth 4096 in
theorem row3067_eq_data : row3067 = Data.row3067 := rfl
def node3067 : Graph := .node 3067 row3067 node2857 node3299
set_option maxRecDepth 4096 in
theorem node3067_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3067 :=
  Budget.node_covers _ _ 3067 row3067 node2857 node3299
    node2857_covers node3299_covers (by decide) (by decide) row3067_covers
theorem node3067_eq_data : node3067 = Data.node3067 := by
  have branches : (node2857, node3299) = (Data.node2857, Data.node3299) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2857_eq_data node3299_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3067 row children.1 children.2) row3067_eq_data branches
set_option maxRecDepth 4096 in
theorem row3491_eq_data : row3491 = Data.row3491 := rfl
def node3491 : Graph := .node 3491 row3491 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3491_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3491 :=
  Budget.node_covers _ _ 3491 row3491 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3491_covers
theorem node3491_eq_data : node3491 = Data.node3491 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3491 row children.1 children.2) row3491_eq_data branches
set_option maxRecDepth 4096 in
theorem row3511_eq_data : row3511 = Data.row3511 := rfl
def node3511 : Graph := .node 3511 row3511 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3511_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3511 :=
  Budget.node_covers _ _ 3511 row3511 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3511_covers
theorem node3511_eq_data : node3511 = Data.node3511 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3511 row children.1 children.2) row3511_eq_data branches
set_option maxRecDepth 4096 in
theorem row3499_eq_data : row3499 = Data.row3499 := rfl
def node3499 : Graph := .node 3499 row3499 node3491 node3511
set_option maxRecDepth 4096 in
theorem node3499_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3499 :=
  Budget.node_covers _ _ 3499 row3499 node3491 node3511
    node3491_covers node3511_covers (by decide) (by decide) row3499_covers
theorem node3499_eq_data : node3499 = Data.node3499 := by
  have branches : (node3491, node3511) = (Data.node3491, Data.node3511) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3491_eq_data node3511_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3499 row children.1 children.2) row3499_eq_data branches
set_option maxRecDepth 4096 in
theorem row3527_eq_data : row3527 = Data.row3527 := rfl
def node3527 : Graph := .node 3527 row3527 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3527_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3527 :=
  Budget.node_covers _ _ 3527 row3527 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3527_covers
theorem node3527_eq_data : node3527 = Data.node3527 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3527 row children.1 children.2) row3527_eq_data branches
set_option maxRecDepth 4096 in
theorem row3529_eq_data : row3529 = Data.row3529 := rfl
def node3529 : Graph := .node 3529 row3529 node3527 Graph.empty
set_option maxRecDepth 4096 in
theorem node3529_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3529 :=
  Budget.node_covers _ _ 3529 row3529 node3527 Graph.empty
    node3527_covers empty_covers (by decide) (by decide) row3529_covers
theorem node3529_eq_data : node3529 = Data.node3529 := by
  have branches : (node3527, Graph.empty) = (Data.node3527, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3527_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3529 row children.1 children.2) row3529_eq_data branches
set_option maxRecDepth 4096 in
theorem row3517_eq_data : row3517 = Data.row3517 := rfl
def node3517 : Graph := .node 3517 row3517 node3499 node3529
set_option maxRecDepth 4096 in
theorem node3517_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3517 :=
  Budget.node_covers _ _ 3517 row3517 node3499 node3529
    node3499_covers node3529_covers (by decide) (by decide) row3517_covers
theorem node3517_eq_data : node3517 = Data.node3517 := by
  have branches : (node3499, node3529) = (Data.node3499, Data.node3529) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3499_eq_data node3529_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3517 row children.1 children.2) row3517_eq_data branches
set_option maxRecDepth 4096 in
theorem row3539_eq_data : row3539 = Data.row3539 := rfl
def node3539 : Graph := .node 3539 row3539 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3539_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3539 :=
  Budget.node_covers _ _ 3539 row3539 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3539_covers
theorem node3539_eq_data : node3539 = Data.node3539 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3539 row children.1 children.2) row3539_eq_data branches
set_option maxRecDepth 4096 in
theorem row3541_eq_data : row3541 = Data.row3541 := rfl
def node3541 : Graph := .node 3541 row3541 node3539 Graph.empty
set_option maxRecDepth 4096 in
theorem node3541_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3541 :=
  Budget.node_covers _ _ 3541 row3541 node3539 Graph.empty
    node3539_covers empty_covers (by decide) (by decide) row3541_covers
theorem node3541_eq_data : node3541 = Data.node3541 := by
  have branches : (node3539, Graph.empty) = (Data.node3539, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3539_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3541 row children.1 children.2) row3541_eq_data branches
set_option maxRecDepth 4096 in
theorem row3557_eq_data : row3557 = Data.row3557 := rfl
def node3557 : Graph := .node 3557 row3557 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3557_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3557 :=
  Budget.node_covers _ _ 3557 row3557 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3557_covers
theorem node3557_eq_data : node3557 = Data.node3557 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3557 row children.1 children.2) row3557_eq_data branches
set_option maxRecDepth 4096 in
theorem row3559_eq_data : row3559 = Data.row3559 := rfl
def node3559 : Graph := .node 3559 row3559 node3557 Graph.empty
set_option maxRecDepth 4096 in
theorem node3559_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3559 :=
  Budget.node_covers _ _ 3559 row3559 node3557 Graph.empty
    node3557_covers empty_covers (by decide) (by decide) row3559_covers
theorem node3559_eq_data : node3559 = Data.node3559 := by
  have branches : (node3557, Graph.empty) = (Data.node3557, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3557_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3559 row children.1 children.2) row3559_eq_data branches
set_option maxRecDepth 4096 in
theorem row3547_eq_data : row3547 = Data.row3547 := rfl
def node3547 : Graph := .node 3547 row3547 node3541 node3559
set_option maxRecDepth 4096 in
theorem node3547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3547 :=
  Budget.node_covers _ _ 3547 row3547 node3541 node3559
    node3541_covers node3559_covers (by decide) (by decide) row3547_covers
theorem node3547_eq_data : node3547 = Data.node3547 := by
  have branches : (node3541, node3559) = (Data.node3541, Data.node3559) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3541_eq_data node3559_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3547 row children.1 children.2) row3547_eq_data branches
set_option maxRecDepth 4096 in
theorem row3533_eq_data : row3533 = Data.row3533 := rfl
def node3533 : Graph := .node 3533 row3533 node3517 node3547
set_option maxRecDepth 4096 in
theorem node3533_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3533 :=
  Budget.node_covers _ _ 3533 row3533 node3517 node3547
    node3517_covers node3547_covers (by decide) (by decide) row3533_covers
theorem node3533_eq_data : node3533 = Data.node3533 := by
  have branches : (node3517, node3547) = (Data.node3517, Data.node3547) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3517_eq_data node3547_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3533 row children.1 children.2) row3533_eq_data branches
set_option maxRecDepth 4096 in
theorem row3581_eq_data : row3581 = Data.row3581 := rfl
def node3581 : Graph := .node 3581 row3581 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3581_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3581 :=
  Budget.node_covers _ _ 3581 row3581 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3581_covers
theorem node3581_eq_data : node3581 = Data.node3581 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3581 row children.1 children.2) row3581_eq_data branches
set_option maxRecDepth 4096 in
theorem row3583_eq_data : row3583 = Data.row3583 := rfl
def node3583 : Graph := .node 3583 row3583 node3581 Graph.empty
set_option maxRecDepth 4096 in
theorem node3583_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3583 :=
  Budget.node_covers _ _ 3583 row3583 node3581 Graph.empty
    node3581_covers empty_covers (by decide) (by decide) row3583_covers
theorem node3583_eq_data : node3583 = Data.node3583 := by
  have branches : (node3581, Graph.empty) = (Data.node3581, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3581_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3583 row children.1 children.2) row3583_eq_data branches
set_option maxRecDepth 4096 in
theorem row3607_eq_data : row3607 = Data.row3607 := rfl
def node3607 : Graph := .node 3607 row3607 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3607_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3607 :=
  Budget.node_covers _ _ 3607 row3607 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3607_covers
theorem node3607_eq_data : node3607 = Data.node3607 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3607 row children.1 children.2) row3607_eq_data branches
set_option maxRecDepth 4096 in
theorem row3613_eq_data : row3613 = Data.row3613 := rfl
def node3613 : Graph := .node 3613 row3613 node3607 Graph.empty
set_option maxRecDepth 4096 in
theorem node3613_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3613 :=
  Budget.node_covers _ _ 3613 row3613 node3607 Graph.empty
    node3607_covers empty_covers (by decide) (by decide) row3613_covers
theorem node3613_eq_data : node3613 = Data.node3613 := by
  have branches : (node3607, Graph.empty) = (Data.node3607, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3607_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3613 row children.1 children.2) row3613_eq_data branches
set_option maxRecDepth 4096 in
theorem row3593_eq_data : row3593 = Data.row3593 := rfl
def node3593 : Graph := .node 3593 row3593 node3583 node3613
set_option maxRecDepth 4096 in
theorem node3593_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3593 :=
  Budget.node_covers _ _ 3593 row3593 node3583 node3613
    node3583_covers node3613_covers (by decide) (by decide) row3593_covers
theorem node3593_eq_data : node3593 = Data.node3593 := by
  have branches : (node3583, node3613) = (Data.node3583, Data.node3613) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3583_eq_data node3613_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3593 row children.1 children.2) row3593_eq_data branches
set_option maxRecDepth 4096 in
theorem row3623_eq_data : row3623 = Data.row3623 := rfl
def node3623 : Graph := .node 3623 row3623 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3623_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3623 :=
  Budget.node_covers _ _ 3623 row3623 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3623_covers
theorem node3623_eq_data : node3623 = Data.node3623 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3623 row children.1 children.2) row3623_eq_data branches
set_option maxRecDepth 4096 in
theorem row3631_eq_data : row3631 = Data.row3631 := rfl
def node3631 : Graph := .node 3631 row3631 node3623 Graph.empty
set_option maxRecDepth 4096 in
theorem node3631_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3631 :=
  Budget.node_covers _ _ 3631 row3631 node3623 Graph.empty
    node3623_covers empty_covers (by decide) (by decide) row3631_covers
theorem node3631_eq_data : node3631 = Data.node3631 := by
  have branches : (node3623, Graph.empty) = (Data.node3623, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3623_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3631 row children.1 children.2) row3631_eq_data branches
set_option maxRecDepth 4096 in
theorem row3643_eq_data : row3643 = Data.row3643 := rfl
def node3643 : Graph := .node 3643 row3643 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3643_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3643 :=
  Budget.node_covers _ _ 3643 row3643 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3643_covers
theorem node3643_eq_data : node3643 = Data.node3643 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3643 row children.1 children.2) row3643_eq_data branches
set_option maxRecDepth 4096 in
theorem row3659_eq_data : row3659 = Data.row3659 := rfl
def node3659 : Graph := .node 3659 row3659 node3643 Graph.empty
set_option maxRecDepth 4096 in
theorem node3659_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3659 :=
  Budget.node_covers _ _ 3659 row3659 node3643 Graph.empty
    node3643_covers empty_covers (by decide) (by decide) row3659_covers
theorem node3659_eq_data : node3659 = Data.node3659 := by
  have branches : (node3643, Graph.empty) = (Data.node3643, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3643_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3659 row children.1 children.2) row3659_eq_data branches
set_option maxRecDepth 4096 in
theorem row3637_eq_data : row3637 = Data.row3637 := rfl
def node3637 : Graph := .node 3637 row3637 node3631 node3659
set_option maxRecDepth 4096 in
theorem node3637_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3637 :=
  Budget.node_covers _ _ 3637 row3637 node3631 node3659
    node3631_covers node3659_covers (by decide) (by decide) row3637_covers
theorem node3637_eq_data : node3637 = Data.node3637 := by
  have branches : (node3631, node3659) = (Data.node3631, Data.node3659) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3631_eq_data node3659_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3637 row children.1 children.2) row3637_eq_data branches
set_option maxRecDepth 4096 in
theorem row3617_eq_data : row3617 = Data.row3617 := rfl
def node3617 : Graph := .node 3617 row3617 node3593 node3637
set_option maxRecDepth 4096 in
theorem node3617_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3617 :=
  Budget.node_covers _ _ 3617 row3617 node3593 node3637
    node3593_covers node3637_covers (by decide) (by decide) row3617_covers
theorem node3617_eq_data : node3617 = Data.node3617 := by
  have branches : (node3593, node3637) = (Data.node3593, Data.node3637) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3593_eq_data node3637_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3617 row children.1 children.2) row3617_eq_data branches
set_option maxRecDepth 4096 in
theorem row3571_eq_data : row3571 = Data.row3571 := rfl
def node3571 : Graph := .node 3571 row3571 node3533 node3617
set_option maxRecDepth 4096 in
theorem node3571_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3571 :=
  Budget.node_covers _ _ 3571 row3571 node3533 node3617
    node3533_covers node3617_covers (by decide) (by decide) row3571_covers
theorem node3571_eq_data : node3571 = Data.node3571 := by
  have branches : (node3533, node3617) = (Data.node3533, Data.node3617) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3533_eq_data node3617_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3571 row children.1 children.2) row3571_eq_data branches
set_option maxRecDepth 4096 in
theorem row3673_eq_data : row3673 = Data.row3673 := rfl
def node3673 : Graph := .node 3673 row3673 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3673_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3673 :=
  Budget.node_covers _ _ 3673 row3673 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3673_covers
theorem node3673_eq_data : node3673 = Data.node3673 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3673 row children.1 children.2) row3673_eq_data branches
set_option maxRecDepth 4096 in
theorem row3677_eq_data : row3677 = Data.row3677 := rfl
def node3677 : Graph := .node 3677 row3677 node3673 Graph.empty
set_option maxRecDepth 4096 in
theorem node3677_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3677 :=
  Budget.node_covers _ _ 3677 row3677 node3673 Graph.empty
    node3673_covers empty_covers (by decide) (by decide) row3677_covers
theorem node3677_eq_data : node3677 = Data.node3677 := by
  have branches : (node3673, Graph.empty) = (Data.node3673, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3673_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3677 row children.1 children.2) row3677_eq_data branches
set_option maxRecDepth 4096 in
theorem row3697_eq_data : row3697 = Data.row3697 := rfl
def node3697 : Graph := .node 3697 row3697 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node3697_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3697 :=
  Budget.node_covers _ _ 3697 row3697 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row3697_covers
theorem node3697_eq_data : node3697 = Data.node3697 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3697 row children.1 children.2) row3697_eq_data branches
set_option maxRecDepth 4096 in
theorem row3701_eq_data : row3701 = Data.row3701 := rfl
def node3701 : Graph := .node 3701 row3701 node3697 Graph.empty
set_option maxRecDepth 4096 in
theorem node3701_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3701 :=
  Budget.node_covers _ _ 3701 row3701 node3697 Graph.empty
    node3697_covers empty_covers (by decide) (by decide) row3701_covers
theorem node3701_eq_data : node3701 = Data.node3701 := by
  have branches : (node3697, Graph.empty) = (Data.node3697, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3697_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3701 row children.1 children.2) row3701_eq_data branches
set_option maxRecDepth 4096 in
theorem row3691_eq_data : row3691 = Data.row3691 := rfl
def node3691 : Graph := .node 3691 row3691 node3677 node3701
set_option maxRecDepth 4096 in
theorem node3691_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3691 :=
  Budget.node_covers _ _ 3691 row3691 node3677 node3701
    node3677_covers node3701_covers (by decide) (by decide) row3691_covers
theorem node3691_eq_data : node3691 = Data.node3691 := by
  have branches : (node3677, node3701) = (Data.node3677, Data.node3701) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3677_eq_data node3701_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3691 row children.1 children.2) row3691_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
