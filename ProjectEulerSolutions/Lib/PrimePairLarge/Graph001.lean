import ProjectEulerSolutions.Lib.PrimePairLarge.Graph000
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row349_eq_data : row349 = Data.row349 := rfl
def node349 : Graph := .node 349 row349 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node349_covers : Budget.Covers PrimePairVertices.largeTree 26034 node349 :=
  Budget.node_covers _ _ 349 row349 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row349_covers
theorem node349_eq_data : node349 = Data.node349 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 349 row children.1 children.2) row349_eq_data branches
set_option maxRecDepth 4096 in
theorem row353_eq_data : row353 = Data.row353 := rfl
def node353 : Graph := .node 353 row353 node349 Graph.empty
set_option maxRecDepth 4096 in
theorem node353_covers : Budget.Covers PrimePairVertices.largeTree 26034 node353 :=
  Budget.node_covers _ _ 353 row353 node349 Graph.empty
    node349_covers empty_covers (by decide) (by decide) row353_covers
theorem node353_eq_data : node353 = Data.node353 := by
  have branches : (node349, Graph.empty) = (Data.node349, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node349_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 353 row children.1 children.2) row353_eq_data branches
set_option maxRecDepth 4096 in
theorem row367_eq_data : row367 = Data.row367 := rfl
def node367 : Graph := .node 367 row367 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node367_covers : Budget.Covers PrimePairVertices.largeTree 26034 node367 :=
  Budget.node_covers _ _ 367 row367 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row367_covers
theorem node367_eq_data : node367 = Data.node367 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 367 row children.1 children.2) row367_eq_data branches
set_option maxRecDepth 4096 in
theorem row373_eq_data : row373 = Data.row373 := rfl
def node373 : Graph := .node 373 row373 node367 Graph.empty
set_option maxRecDepth 4096 in
theorem node373_covers : Budget.Covers PrimePairVertices.largeTree 26034 node373 :=
  Budget.node_covers _ _ 373 row373 node367 Graph.empty
    node367_covers empty_covers (by decide) (by decide) row373_covers
theorem node373_eq_data : node373 = Data.node373 := by
  have branches : (node367, Graph.empty) = (Data.node367, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node367_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 373 row children.1 children.2) row373_eq_data branches
set_option maxRecDepth 4096 in
theorem row359_eq_data : row359 = Data.row359 := rfl
def node359 : Graph := .node 359 row359 node353 node373
set_option maxRecDepth 4096 in
theorem node359_covers : Budget.Covers PrimePairVertices.largeTree 26034 node359 :=
  Budget.node_covers _ _ 359 row359 node353 node373
    node353_covers node373_covers (by decide) (by decide) row359_covers
theorem node359_eq_data : node359 = Data.node359 := by
  have branches : (node353, node373) = (Data.node353, Data.node373) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node353_eq_data node373_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 359 row children.1 children.2) row359_eq_data branches
set_option maxRecDepth 4096 in
theorem row347_eq_data : row347 = Data.row347 := rfl
def node347 : Graph := .node 347 row347 node317 node359
set_option maxRecDepth 4096 in
theorem node347_covers : Budget.Covers PrimePairVertices.largeTree 26034 node347 :=
  Budget.node_covers _ _ 347 row347 node317 node359
    node317_covers node359_covers (by decide) (by decide) row347_covers
theorem node347_eq_data : node347 = Data.node347 := by
  have branches : (node317, node359) = (Data.node317, Data.node359) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node317_eq_data node359_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 347 row children.1 children.2) row347_eq_data branches
set_option maxRecDepth 4096 in
theorem row307_eq_data : row307 = Data.row307 := rfl
def node307 : Graph := .node 307 row307 node269 node347
set_option maxRecDepth 4096 in
theorem node307_covers : Budget.Covers PrimePairVertices.largeTree 26034 node307 :=
  Budget.node_covers _ _ 307 row307 node269 node347
    node269_covers node347_covers (by decide) (by decide) row307_covers
theorem node307_eq_data : node307 = Data.node307 := by
  have branches : (node269, node347) = (Data.node269, Data.node347) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node269_eq_data node347_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 307 row children.1 children.2) row307_eq_data branches
set_option maxRecDepth 4096 in
theorem row383_eq_data : row383 = Data.row383 := rfl
def node383 : Graph := .node 383 row383 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node383_covers : Budget.Covers PrimePairVertices.largeTree 26034 node383 :=
  Budget.node_covers _ _ 383 row383 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row383_covers
theorem node383_eq_data : node383 = Data.node383 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 383 row children.1 children.2) row383_eq_data branches
set_option maxRecDepth 4096 in
theorem row389_eq_data : row389 = Data.row389 := rfl
def node389 : Graph := .node 389 row389 node383 Graph.empty
set_option maxRecDepth 4096 in
theorem node389_covers : Budget.Covers PrimePairVertices.largeTree 26034 node389 :=
  Budget.node_covers _ _ 389 row389 node383 Graph.empty
    node383_covers empty_covers (by decide) (by decide) row389_covers
theorem node389_eq_data : node389 = Data.node389 := by
  have branches : (node383, Graph.empty) = (Data.node383, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node383_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 389 row children.1 children.2) row389_eq_data branches
set_option maxRecDepth 4096 in
theorem row401_eq_data : row401 = Data.row401 := rfl
def node401 : Graph := .node 401 row401 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node401_covers : Budget.Covers PrimePairVertices.largeTree 26034 node401 :=
  Budget.node_covers _ _ 401 row401 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row401_covers
theorem node401_eq_data : node401 = Data.node401 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 401 row children.1 children.2) row401_eq_data branches
set_option maxRecDepth 4096 in
theorem row409_eq_data : row409 = Data.row409 := rfl
def node409 : Graph := .node 409 row409 node401 Graph.empty
set_option maxRecDepth 4096 in
theorem node409_covers : Budget.Covers PrimePairVertices.largeTree 26034 node409 :=
  Budget.node_covers _ _ 409 row409 node401 Graph.empty
    node401_covers empty_covers (by decide) (by decide) row409_covers
theorem node409_eq_data : node409 = Data.node409 := by
  have branches : (node401, Graph.empty) = (Data.node401, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node401_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 409 row children.1 children.2) row409_eq_data branches
set_option maxRecDepth 4096 in
theorem row397_eq_data : row397 = Data.row397 := rfl
def node397 : Graph := .node 397 row397 node389 node409
set_option maxRecDepth 4096 in
theorem node397_covers : Budget.Covers PrimePairVertices.largeTree 26034 node397 :=
  Budget.node_covers _ _ 397 row397 node389 node409
    node389_covers node409_covers (by decide) (by decide) row397_covers
theorem node397_eq_data : node397 = Data.node397 := by
  have branches : (node389, node409) = (Data.node389, Data.node409) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node389_eq_data node409_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 397 row children.1 children.2) row397_eq_data branches
set_option maxRecDepth 4096 in
theorem row421_eq_data : row421 = Data.row421 := rfl
def node421 : Graph := .node 421 row421 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node421_covers : Budget.Covers PrimePairVertices.largeTree 26034 node421 :=
  Budget.node_covers _ _ 421 row421 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row421_covers
theorem node421_eq_data : node421 = Data.node421 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 421 row children.1 children.2) row421_eq_data branches
set_option maxRecDepth 4096 in
theorem row431_eq_data : row431 = Data.row431 := rfl
def node431 : Graph := .node 431 row431 node421 Graph.empty
set_option maxRecDepth 4096 in
theorem node431_covers : Budget.Covers PrimePairVertices.largeTree 26034 node431 :=
  Budget.node_covers _ _ 431 row431 node421 Graph.empty
    node421_covers empty_covers (by decide) (by decide) row431_covers
theorem node431_eq_data : node431 = Data.node431 := by
  have branches : (node421, Graph.empty) = (Data.node421, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node421_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 431 row children.1 children.2) row431_eq_data branches
set_option maxRecDepth 4096 in
theorem row439_eq_data : row439 = Data.row439 := rfl
def node439 : Graph := .node 439 row439 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node439_covers : Budget.Covers PrimePairVertices.largeTree 26034 node439 :=
  Budget.node_covers _ _ 439 row439 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row439_covers
theorem node439_eq_data : node439 = Data.node439 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 439 row children.1 children.2) row439_eq_data branches
set_option maxRecDepth 4096 in
theorem row443_eq_data : row443 = Data.row443 := rfl
def node443 : Graph := .node 443 row443 node439 Graph.empty
set_option maxRecDepth 4096 in
theorem node443_covers : Budget.Covers PrimePairVertices.largeTree 26034 node443 :=
  Budget.node_covers _ _ 443 row443 node439 Graph.empty
    node439_covers empty_covers (by decide) (by decide) row443_covers
theorem node443_eq_data : node443 = Data.node443 := by
  have branches : (node439, Graph.empty) = (Data.node439, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node439_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 443 row children.1 children.2) row443_eq_data branches
set_option maxRecDepth 4096 in
theorem row433_eq_data : row433 = Data.row433 := rfl
def node433 : Graph := .node 433 row433 node431 node443
set_option maxRecDepth 4096 in
theorem node433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node433 :=
  Budget.node_covers _ _ 433 row433 node431 node443
    node431_covers node443_covers (by decide) (by decide) row433_covers
theorem node433_eq_data : node433 = Data.node433 := by
  have branches : (node431, node443) = (Data.node431, Data.node443) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node431_eq_data node443_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 433 row children.1 children.2) row433_eq_data branches
set_option maxRecDepth 4096 in
theorem row419_eq_data : row419 = Data.row419 := rfl
def node419 : Graph := .node 419 row419 node397 node433
set_option maxRecDepth 4096 in
theorem node419_covers : Budget.Covers PrimePairVertices.largeTree 26034 node419 :=
  Budget.node_covers _ _ 419 row419 node397 node433
    node397_covers node433_covers (by decide) (by decide) row419_covers
theorem node419_eq_data : node419 = Data.node419 := by
  have branches : (node397, node433) = (Data.node397, Data.node433) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node397_eq_data node433_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 419 row children.1 children.2) row419_eq_data branches
set_option maxRecDepth 4096 in
theorem row457_eq_data : row457 = Data.row457 := rfl
def node457 : Graph := .node 457 row457 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node457_covers : Budget.Covers PrimePairVertices.largeTree 26034 node457 :=
  Budget.node_covers _ _ 457 row457 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row457_covers
theorem node457_eq_data : node457 = Data.node457 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 457 row children.1 children.2) row457_eq_data branches
set_option maxRecDepth 4096 in
theorem row461_eq_data : row461 = Data.row461 := rfl
def node461 : Graph := .node 461 row461 node457 Graph.empty
set_option maxRecDepth 4096 in
theorem node461_covers : Budget.Covers PrimePairVertices.largeTree 26034 node461 :=
  Budget.node_covers _ _ 461 row461 node457 Graph.empty
    node457_covers empty_covers (by decide) (by decide) row461_covers
theorem node461_eq_data : node461 = Data.node461 := by
  have branches : (node457, Graph.empty) = (Data.node457, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node457_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 461 row children.1 children.2) row461_eq_data branches
set_option maxRecDepth 4096 in
theorem row467_eq_data : row467 = Data.row467 := rfl
def node467 : Graph := .node 467 row467 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node467_covers : Budget.Covers PrimePairVertices.largeTree 26034 node467 :=
  Budget.node_covers _ _ 467 row467 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row467_covers
theorem node467_eq_data : node467 = Data.node467 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 467 row children.1 children.2) row467_eq_data branches
set_option maxRecDepth 4096 in
theorem row479_eq_data : row479 = Data.row479 := rfl
def node479 : Graph := .node 479 row479 node467 Graph.empty
set_option maxRecDepth 4096 in
theorem node479_covers : Budget.Covers PrimePairVertices.largeTree 26034 node479 :=
  Budget.node_covers _ _ 479 row479 node467 Graph.empty
    node467_covers empty_covers (by decide) (by decide) row479_covers
theorem node479_eq_data : node479 = Data.node479 := by
  have branches : (node467, Graph.empty) = (Data.node467, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node467_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 479 row children.1 children.2) row479_eq_data branches
set_option maxRecDepth 4096 in
theorem row463_eq_data : row463 = Data.row463 := rfl
def node463 : Graph := .node 463 row463 node461 node479
set_option maxRecDepth 4096 in
theorem node463_covers : Budget.Covers PrimePairVertices.largeTree 26034 node463 :=
  Budget.node_covers _ _ 463 row463 node461 node479
    node461_covers node479_covers (by decide) (by decide) row463_covers
theorem node463_eq_data : node463 = Data.node463 := by
  have branches : (node461, node479) = (Data.node461, Data.node479) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node461_eq_data node479_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 463 row children.1 children.2) row463_eq_data branches
set_option maxRecDepth 4096 in
theorem row491_eq_data : row491 = Data.row491 := rfl
def node491 : Graph := .node 491 row491 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node491_covers : Budget.Covers PrimePairVertices.largeTree 26034 node491 :=
  Budget.node_covers _ _ 491 row491 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row491_covers
theorem node491_eq_data : node491 = Data.node491 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 491 row children.1 children.2) row491_eq_data branches
set_option maxRecDepth 4096 in
theorem row499_eq_data : row499 = Data.row499 := rfl
def node499 : Graph := .node 499 row499 node491 Graph.empty
set_option maxRecDepth 4096 in
theorem node499_covers : Budget.Covers PrimePairVertices.largeTree 26034 node499 :=
  Budget.node_covers _ _ 499 row499 node491 Graph.empty
    node491_covers empty_covers (by decide) (by decide) row499_covers
theorem node499_eq_data : node499 = Data.node499 := by
  have branches : (node491, Graph.empty) = (Data.node491, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node491_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 499 row children.1 children.2) row499_eq_data branches
set_option maxRecDepth 4096 in
theorem row509_eq_data : row509 = Data.row509 := rfl
def node509 : Graph := .node 509 row509 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node509_covers : Budget.Covers PrimePairVertices.largeTree 26034 node509 :=
  Budget.node_covers _ _ 509 row509 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row509_covers
theorem node509_eq_data : node509 = Data.node509 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 509 row children.1 children.2) row509_eq_data branches
set_option maxRecDepth 4096 in
theorem row521_eq_data : row521 = Data.row521 := rfl
def node521 : Graph := .node 521 row521 node509 Graph.empty
set_option maxRecDepth 4096 in
theorem node521_covers : Budget.Covers PrimePairVertices.largeTree 26034 node521 :=
  Budget.node_covers _ _ 521 row521 node509 Graph.empty
    node509_covers empty_covers (by decide) (by decide) row521_covers
theorem node521_eq_data : node521 = Data.node521 := by
  have branches : (node509, Graph.empty) = (Data.node509, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node509_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 521 row children.1 children.2) row521_eq_data branches
set_option maxRecDepth 4096 in
theorem row503_eq_data : row503 = Data.row503 := rfl
def node503 : Graph := .node 503 row503 node499 node521
set_option maxRecDepth 4096 in
theorem node503_covers : Budget.Covers PrimePairVertices.largeTree 26034 node503 :=
  Budget.node_covers _ _ 503 row503 node499 node521
    node499_covers node521_covers (by decide) (by decide) row503_covers
theorem node503_eq_data : node503 = Data.node503 := by
  have branches : (node499, node521) = (Data.node499, Data.node521) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node499_eq_data node521_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 503 row children.1 children.2) row503_eq_data branches
set_option maxRecDepth 4096 in
theorem row487_eq_data : row487 = Data.row487 := rfl
def node487 : Graph := .node 487 row487 node463 node503
set_option maxRecDepth 4096 in
theorem node487_covers : Budget.Covers PrimePairVertices.largeTree 26034 node487 :=
  Budget.node_covers _ _ 487 row487 node463 node503
    node463_covers node503_covers (by decide) (by decide) row487_covers
theorem node487_eq_data : node487 = Data.node487 := by
  have branches : (node463, node503) = (Data.node463, Data.node503) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node463_eq_data node503_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 487 row children.1 children.2) row487_eq_data branches
set_option maxRecDepth 4096 in
theorem row449_eq_data : row449 = Data.row449 := rfl
def node449 : Graph := .node 449 row449 node419 node487
set_option maxRecDepth 4096 in
theorem node449_covers : Budget.Covers PrimePairVertices.largeTree 26034 node449 :=
  Budget.node_covers _ _ 449 row449 node419 node487
    node419_covers node487_covers (by decide) (by decide) row449_covers
theorem node449_eq_data : node449 = Data.node449 := by
  have branches : (node419, node487) = (Data.node419, Data.node487) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node419_eq_data node487_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 449 row children.1 children.2) row449_eq_data branches
set_option maxRecDepth 4096 in
theorem row379_eq_data : row379 = Data.row379 := rfl
def node379 : Graph := .node 379 row379 node307 node449
set_option maxRecDepth 4096 in
theorem node379_covers : Budget.Covers PrimePairVertices.largeTree 26034 node379 :=
  Budget.node_covers _ _ 379 row379 node307 node449
    node307_covers node449_covers (by decide) (by decide) row379_covers
theorem node379_eq_data : node379 = Data.node379 := by
  have branches : (node307, node449) = (Data.node307, Data.node449) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node307_eq_data node449_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 379 row children.1 children.2) row379_eq_data branches
set_option maxRecDepth 4096 in
theorem row233_eq_data : row233 = Data.row233 := rfl
def node233 : Graph := .node 233 row233 node103 node379
set_option maxRecDepth 4096 in
theorem node233_covers : Budget.Covers PrimePairVertices.largeTree 26034 node233 :=
  Budget.node_covers _ _ 233 row233 node103 node379
    node103_covers node379_covers (by decide) (by decide) row233_covers
theorem node233_eq_data : node233 = Data.node233 := by
  have branches : (node103, node379) = (Data.node103, Data.node379) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node103_eq_data node379_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 233 row children.1 children.2) row233_eq_data branches
set_option maxRecDepth 4096 in
theorem row541_eq_data : row541 = Data.row541 := rfl
def node541 : Graph := .node 541 row541 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node541_covers : Budget.Covers PrimePairVertices.largeTree 26034 node541 :=
  Budget.node_covers _ _ 541 row541 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row541_covers
theorem node541_eq_data : node541 = Data.node541 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 541 row children.1 children.2) row541_eq_data branches
set_option maxRecDepth 4096 in
theorem row557_eq_data : row557 = Data.row557 := rfl
def node557 : Graph := .node 557 row557 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node557_covers : Budget.Covers PrimePairVertices.largeTree 26034 node557 :=
  Budget.node_covers _ _ 557 row557 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row557_covers
theorem node557_eq_data : node557 = Data.node557 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 557 row children.1 children.2) row557_eq_data branches
set_option maxRecDepth 4096 in
theorem row547_eq_data : row547 = Data.row547 := rfl
def node547 : Graph := .node 547 row547 node541 node557
set_option maxRecDepth 4096 in
theorem node547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node547 :=
  Budget.node_covers _ _ 547 row547 node541 node557
    node541_covers node557_covers (by decide) (by decide) row547_covers
theorem node547_eq_data : node547 = Data.node547 := by
  have branches : (node541, node557) = (Data.node541, Data.node557) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node541_eq_data node557_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 547 row children.1 children.2) row547_eq_data branches
set_option maxRecDepth 4096 in
theorem row569_eq_data : row569 = Data.row569 := rfl
def node569 : Graph := .node 569 row569 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node569_covers : Budget.Covers PrimePairVertices.largeTree 26034 node569 :=
  Budget.node_covers _ _ 569 row569 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row569_covers
theorem node569_eq_data : node569 = Data.node569 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 569 row children.1 children.2) row569_eq_data branches
set_option maxRecDepth 4096 in
theorem row571_eq_data : row571 = Data.row571 := rfl
def node571 : Graph := .node 571 row571 node569 Graph.empty
set_option maxRecDepth 4096 in
theorem node571_covers : Budget.Covers PrimePairVertices.largeTree 26034 node571 :=
  Budget.node_covers _ _ 571 row571 node569 Graph.empty
    node569_covers empty_covers (by decide) (by decide) row571_covers
theorem node571_eq_data : node571 = Data.node571 := by
  have branches : (node569, Graph.empty) = (Data.node569, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node569_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 571 row children.1 children.2) row571_eq_data branches
set_option maxRecDepth 4096 in
theorem row563_eq_data : row563 = Data.row563 := rfl
def node563 : Graph := .node 563 row563 node547 node571
set_option maxRecDepth 4096 in
theorem node563_covers : Budget.Covers PrimePairVertices.largeTree 26034 node563 :=
  Budget.node_covers _ _ 563 row563 node547 node571
    node547_covers node571_covers (by decide) (by decide) row563_covers
theorem node563_eq_data : node563 = Data.node563 := by
  have branches : (node547, node571) = (Data.node547, Data.node571) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node547_eq_data node571_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 563 row children.1 children.2) row563_eq_data branches
set_option maxRecDepth 4096 in
theorem row587_eq_data : row587 = Data.row587 := rfl
def node587 : Graph := .node 587 row587 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node587_covers : Budget.Covers PrimePairVertices.largeTree 26034 node587 :=
  Budget.node_covers _ _ 587 row587 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row587_covers
theorem node587_eq_data : node587 = Data.node587 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 587 row children.1 children.2) row587_eq_data branches
set_option maxRecDepth 4096 in
theorem row593_eq_data : row593 = Data.row593 := rfl
def node593 : Graph := .node 593 row593 node587 Graph.empty
set_option maxRecDepth 4096 in
theorem node593_covers : Budget.Covers PrimePairVertices.largeTree 26034 node593 :=
  Budget.node_covers _ _ 593 row593 node587 Graph.empty
    node587_covers empty_covers (by decide) (by decide) row593_covers
theorem node593_eq_data : node593 = Data.node593 := by
  have branches : (node587, Graph.empty) = (Data.node587, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node587_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 593 row children.1 children.2) row593_eq_data branches
set_option maxRecDepth 4096 in
theorem row601_eq_data : row601 = Data.row601 := rfl
def node601 : Graph := .node 601 row601 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node601_covers : Budget.Covers PrimePairVertices.largeTree 26034 node601 :=
  Budget.node_covers _ _ 601 row601 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row601_covers
theorem node601_eq_data : node601 = Data.node601 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 601 row children.1 children.2) row601_eq_data branches
set_option maxRecDepth 4096 in
theorem row607_eq_data : row607 = Data.row607 := rfl
def node607 : Graph := .node 607 row607 node601 Graph.empty
set_option maxRecDepth 4096 in
theorem node607_covers : Budget.Covers PrimePairVertices.largeTree 26034 node607 :=
  Budget.node_covers _ _ 607 row607 node601 Graph.empty
    node601_covers empty_covers (by decide) (by decide) row607_covers
theorem node607_eq_data : node607 = Data.node607 := by
  have branches : (node601, Graph.empty) = (Data.node601, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node601_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 607 row children.1 children.2) row607_eq_data branches
set_option maxRecDepth 4096 in
theorem row599_eq_data : row599 = Data.row599 := rfl
def node599 : Graph := .node 599 row599 node593 node607
set_option maxRecDepth 4096 in
theorem node599_covers : Budget.Covers PrimePairVertices.largeTree 26034 node599 :=
  Budget.node_covers _ _ 599 row599 node593 node607
    node593_covers node607_covers (by decide) (by decide) row599_covers
theorem node599_eq_data : node599 = Data.node599 := by
  have branches : (node593, node607) = (Data.node593, Data.node607) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node593_eq_data node607_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 599 row children.1 children.2) row599_eq_data branches
set_option maxRecDepth 4096 in
theorem row577_eq_data : row577 = Data.row577 := rfl
def node577 : Graph := .node 577 row577 node563 node599
set_option maxRecDepth 4096 in
theorem node577_covers : Budget.Covers PrimePairVertices.largeTree 26034 node577 :=
  Budget.node_covers _ _ 577 row577 node563 node599
    node563_covers node599_covers (by decide) (by decide) row577_covers
theorem node577_eq_data : node577 = Data.node577 := by
  have branches : (node563, node599) = (Data.node563, Data.node599) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node563_eq_data node599_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 577 row children.1 children.2) row577_eq_data branches
set_option maxRecDepth 4096 in
theorem row617_eq_data : row617 = Data.row617 := rfl
def node617 : Graph := .node 617 row617 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node617_covers : Budget.Covers PrimePairVertices.largeTree 26034 node617 :=
  Budget.node_covers _ _ 617 row617 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row617_covers
theorem node617_eq_data : node617 = Data.node617 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 617 row children.1 children.2) row617_eq_data branches
set_option maxRecDepth 4096 in
theorem row619_eq_data : row619 = Data.row619 := rfl
def node619 : Graph := .node 619 row619 node617 Graph.empty
set_option maxRecDepth 4096 in
theorem node619_covers : Budget.Covers PrimePairVertices.largeTree 26034 node619 :=
  Budget.node_covers _ _ 619 row619 node617 Graph.empty
    node617_covers empty_covers (by decide) (by decide) row619_covers
theorem node619_eq_data : node619 = Data.node619 := by
  have branches : (node617, Graph.empty) = (Data.node617, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node617_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 619 row children.1 children.2) row619_eq_data branches
set_option maxRecDepth 4096 in
theorem row641_eq_data : row641 = Data.row641 := rfl
def node641 : Graph := .node 641 row641 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node641_covers : Budget.Covers PrimePairVertices.largeTree 26034 node641 :=
  Budget.node_covers _ _ 641 row641 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row641_covers
theorem node641_eq_data : node641 = Data.node641 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 641 row children.1 children.2) row641_eq_data branches
set_option maxRecDepth 4096 in
theorem row643_eq_data : row643 = Data.row643 := rfl
def node643 : Graph := .node 643 row643 node641 Graph.empty
set_option maxRecDepth 4096 in
theorem node643_covers : Budget.Covers PrimePairVertices.largeTree 26034 node643 :=
  Budget.node_covers _ _ 643 row643 node641 Graph.empty
    node641_covers empty_covers (by decide) (by decide) row643_covers
theorem node643_eq_data : node643 = Data.node643 := by
  have branches : (node641, Graph.empty) = (Data.node641, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node641_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 643 row children.1 children.2) row643_eq_data branches
set_option maxRecDepth 4096 in
theorem row631_eq_data : row631 = Data.row631 := rfl
def node631 : Graph := .node 631 row631 node619 node643
set_option maxRecDepth 4096 in
theorem node631_covers : Budget.Covers PrimePairVertices.largeTree 26034 node631 :=
  Budget.node_covers _ _ 631 row631 node619 node643
    node619_covers node643_covers (by decide) (by decide) row631_covers
theorem node631_eq_data : node631 = Data.node631 := by
  have branches : (node619, node643) = (Data.node619, Data.node643) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node619_eq_data node643_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 631 row children.1 children.2) row631_eq_data branches
set_option maxRecDepth 4096 in
theorem row653_eq_data : row653 = Data.row653 := rfl
def node653 : Graph := .node 653 row653 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node653_covers : Budget.Covers PrimePairVertices.largeTree 26034 node653 :=
  Budget.node_covers _ _ 653 row653 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row653_covers
theorem node653_eq_data : node653 = Data.node653 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 653 row children.1 children.2) row653_eq_data branches
set_option maxRecDepth 4096 in
theorem row659_eq_data : row659 = Data.row659 := rfl
def node659 : Graph := .node 659 row659 node653 Graph.empty
set_option maxRecDepth 4096 in
theorem node659_covers : Budget.Covers PrimePairVertices.largeTree 26034 node659 :=
  Budget.node_covers _ _ 659 row659 node653 Graph.empty
    node653_covers empty_covers (by decide) (by decide) row659_covers
theorem node659_eq_data : node659 = Data.node659 := by
  have branches : (node653, Graph.empty) = (Data.node653, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node653_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 659 row children.1 children.2) row659_eq_data branches
set_option maxRecDepth 4096 in
theorem row673_eq_data : row673 = Data.row673 := rfl
def node673 : Graph := .node 673 row673 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node673_covers : Budget.Covers PrimePairVertices.largeTree 26034 node673 :=
  Budget.node_covers _ _ 673 row673 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row673_covers
theorem node673_eq_data : node673 = Data.node673 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 673 row children.1 children.2) row673_eq_data branches
set_option maxRecDepth 4096 in
theorem row677_eq_data : row677 = Data.row677 := rfl
def node677 : Graph := .node 677 row677 node673 Graph.empty
set_option maxRecDepth 4096 in
theorem node677_covers : Budget.Covers PrimePairVertices.largeTree 26034 node677 :=
  Budget.node_covers _ _ 677 row677 node673 Graph.empty
    node673_covers empty_covers (by decide) (by decide) row677_covers
theorem node677_eq_data : node677 = Data.node677 := by
  have branches : (node673, Graph.empty) = (Data.node673, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node673_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 677 row children.1 children.2) row677_eq_data branches
set_option maxRecDepth 4096 in
theorem row661_eq_data : row661 = Data.row661 := rfl
def node661 : Graph := .node 661 row661 node659 node677
set_option maxRecDepth 4096 in
theorem node661_covers : Budget.Covers PrimePairVertices.largeTree 26034 node661 :=
  Budget.node_covers _ _ 661 row661 node659 node677
    node659_covers node677_covers (by decide) (by decide) row661_covers
theorem node661_eq_data : node661 = Data.node661 := by
  have branches : (node659, node677) = (Data.node659, Data.node677) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node659_eq_data node677_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 661 row children.1 children.2) row661_eq_data branches
set_option maxRecDepth 4096 in
theorem row647_eq_data : row647 = Data.row647 := rfl
def node647 : Graph := .node 647 row647 node631 node661
set_option maxRecDepth 4096 in
theorem node647_covers : Budget.Covers PrimePairVertices.largeTree 26034 node647 :=
  Budget.node_covers _ _ 647 row647 node631 node661
    node631_covers node661_covers (by decide) (by decide) row647_covers
theorem node647_eq_data : node647 = Data.node647 := by
  have branches : (node631, node661) = (Data.node631, Data.node661) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node631_eq_data node661_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 647 row children.1 children.2) row647_eq_data branches
set_option maxRecDepth 4096 in
theorem row613_eq_data : row613 = Data.row613 := rfl
def node613 : Graph := .node 613 row613 node577 node647
set_option maxRecDepth 4096 in
theorem node613_covers : Budget.Covers PrimePairVertices.largeTree 26034 node613 :=
  Budget.node_covers _ _ 613 row613 node577 node647
    node577_covers node647_covers (by decide) (by decide) row613_covers
theorem node613_eq_data : node613 = Data.node613 := by
  have branches : (node577, node647) = (Data.node577, Data.node647) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node577_eq_data node647_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 613 row children.1 children.2) row613_eq_data branches
set_option maxRecDepth 4096 in
theorem row691_eq_data : row691 = Data.row691 := rfl
def node691 : Graph := .node 691 row691 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node691_covers : Budget.Covers PrimePairVertices.largeTree 26034 node691 :=
  Budget.node_covers _ _ 691 row691 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row691_covers
theorem node691_eq_data : node691 = Data.node691 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 691 row children.1 children.2) row691_eq_data branches
set_option maxRecDepth 4096 in
theorem row701_eq_data : row701 = Data.row701 := rfl
def node701 : Graph := .node 701 row701 node691 Graph.empty
set_option maxRecDepth 4096 in
theorem node701_covers : Budget.Covers PrimePairVertices.largeTree 26034 node701 :=
  Budget.node_covers _ _ 701 row701 node691 Graph.empty
    node691_covers empty_covers (by decide) (by decide) row701_covers
theorem node701_eq_data : node701 = Data.node701 := by
  have branches : (node691, Graph.empty) = (Data.node691, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node691_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 701 row children.1 children.2) row701_eq_data branches
set_option maxRecDepth 4096 in
theorem row719_eq_data : row719 = Data.row719 := rfl
def node719 : Graph := .node 719 row719 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node719 :=
  Budget.node_covers _ _ 719 row719 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row719_covers
theorem node719_eq_data : node719 = Data.node719 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 719 row children.1 children.2) row719_eq_data branches
set_option maxRecDepth 4096 in
theorem row727_eq_data : row727 = Data.row727 := rfl
def node727 : Graph := .node 727 row727 node719 Graph.empty
set_option maxRecDepth 4096 in
theorem node727_covers : Budget.Covers PrimePairVertices.largeTree 26034 node727 :=
  Budget.node_covers _ _ 727 row727 node719 Graph.empty
    node719_covers empty_covers (by decide) (by decide) row727_covers
theorem node727_eq_data : node727 = Data.node727 := by
  have branches : (node719, Graph.empty) = (Data.node719, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node719_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 727 row children.1 children.2) row727_eq_data branches
set_option maxRecDepth 4096 in
theorem row709_eq_data : row709 = Data.row709 := rfl
def node709 : Graph := .node 709 row709 node701 node727
set_option maxRecDepth 4096 in
theorem node709_covers : Budget.Covers PrimePairVertices.largeTree 26034 node709 :=
  Budget.node_covers _ _ 709 row709 node701 node727
    node701_covers node727_covers (by decide) (by decide) row709_covers
theorem node709_eq_data : node709 = Data.node709 := by
  have branches : (node701, node727) = (Data.node701, Data.node727) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node701_eq_data node727_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 709 row children.1 children.2) row709_eq_data branches
set_option maxRecDepth 4096 in
theorem row739_eq_data : row739 = Data.row739 := rfl
def node739 : Graph := .node 739 row739 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node739_covers : Budget.Covers PrimePairVertices.largeTree 26034 node739 :=
  Budget.node_covers _ _ 739 row739 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row739_covers
theorem node739_eq_data : node739 = Data.node739 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 739 row children.1 children.2) row739_eq_data branches
set_option maxRecDepth 4096 in
theorem row743_eq_data : row743 = Data.row743 := rfl
def node743 : Graph := .node 743 row743 node739 Graph.empty
set_option maxRecDepth 4096 in
theorem node743_covers : Budget.Covers PrimePairVertices.largeTree 26034 node743 :=
  Budget.node_covers _ _ 743 row743 node739 Graph.empty
    node739_covers empty_covers (by decide) (by decide) row743_covers
theorem node743_eq_data : node743 = Data.node743 := by
  have branches : (node739, Graph.empty) = (Data.node739, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node739_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 743 row children.1 children.2) row743_eq_data branches
set_option maxRecDepth 4096 in
theorem row757_eq_data : row757 = Data.row757 := rfl
def node757 : Graph := .node 757 row757 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node757_covers : Budget.Covers PrimePairVertices.largeTree 26034 node757 :=
  Budget.node_covers _ _ 757 row757 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row757_covers
theorem node757_eq_data : node757 = Data.node757 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 757 row children.1 children.2) row757_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
