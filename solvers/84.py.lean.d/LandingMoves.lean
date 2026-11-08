import solvers.«84.py.lean.d».LandingBackThree
import solvers.«84.py.lean.d».LandingWeightPositive

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false
set_option maxRecDepth 2048

def moveBody : List Stmt := [.ifThen (.compare (.name "destination") [(.eq, .constant (.int 33))])
  [backLoop] [.assign [.name "outcome"] (.tuple [.name "destination", .constant (.bool false)]), increment (.constant (.int 16))]]

def moveData (destination : Int) : List CardData :=
  if destination = 33 then backData else [((destination, false), 16)]

theorem move_source : (pool.branch 2)[(1 : Nat)]?.bind (fun statement => match statement with
    | .forIn _ _ body _ => some body
    | _ => none) = some moveBody := rfl

theorem bind_weights (cards : List Card) (env : HeapCalls.Env) :
    cards.foldl bind env "weights" = env "weights" := by
  induction cards generalizing env with
  | nil => rfl
  | cons card rest ih => simp [List.foldl_cons, ih, bind, HeapCalls.assignNames, HeapCalls.setLocal]

theorem valid_cards (cards : List Card) (heap : Mutable.Heap) (entries : List Entry)
    (valid : Valid heap entries) (views : ∀ card ∈ cards, card.View heap) :
    Valid heap (cards.foldl applyCard entries) := by
  induction cards generalizing entries with
  | nil => exact valid
  | cons card rest ih =>
      exact ih (applyCard entries card) (valid.inserted _ _ _ (views card (by simp)).2)
        (fun item member => views item (by simp [member]))

theorem distinct_cards (cards : List Card) (entries : List Entry) (distinct : Distinct entries) :
    Distinct (cards.foldl applyCard entries) := by
  induction cards generalizing entries with
  | nil => exact distinct
  | cons card rest ih => exact ih (applyCard entries card) (distinct.added _ _ _)

theorem positive_cards (cards : List Card) (entries : List Entry) (positive : Positive entries)
    (pos : ∀ card ∈ cards, 0 < card.entry.weight) : Positive (cards.foldl applyCard entries) := by
  induction cards generalizing entries with
  | nil => exact positive
  | cons card rest ih =>
      exact ih (applyCard entries card) (positive.added _ _ _ (pos card (by simp)))
        (fun item member => pos item (by simp [member]))

/-- Each Chance movement adds its exact card contribution and touches only its private dictionary. -/
theorem move_continue (destination : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root : Nat) (entries : List Entry) (found : heap[root]? = some (.keyDict (encode entries)))
    (valid : Valid heap entries) (distinct : Distinct entries) (positive : Positive entries)
    (getWeights : env "weights" = some (.ref root))
    (getDestination : env "destination" = some (.scalar (.int destination)))
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : ∀ (nextEnv : HeapCalls.Env) (nextEntries : List Entry) (extra : Mutable.Heap),
      nextEnv "weights" = some (.ref root) →
      Valid ((heap ++ extra).set root (.keyDict (encode nextEntries))) nextEntries →
      Distinct nextEntries →
      Positive nextEntries →
      (∀ landing, (lookup nextEntries landing).getD 0 = (lookup entries landing).getD 0 + total (moveData destination) landing) →
      HeapMemoCalls.BlockReturnsRel program pool.spec nextEnv body
        ((heap ++ extra).set root (.keyDict (encode nextEntries))) post globals) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env (moveBody ++ body) heap post globals := by
  have tested : HeapMemoCalls.Evaluates program pool.spec env
      (.compare (.name "destination") [(.eq, .constant (.int 33))]) heap
      (.scalar (.bool (decide (destination = 33)))) heap globals := by
    apply HeapMemoCalls.Evaluates.compare (HeapMemoCalls.Evaluates.lookup getDestination) HeapMemoCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  apply HeapMemoCalls.BlockReturnsRel.branch tested.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨decide (destination = 33), rfl, ?_⟩
  by_cases community : destination = 33
  · simp only [community, decide_true, ite_true]
    apply back_continue env heap globals root entries found getWeights body post
    intro cards extra correspondence views
    apply rest (cards.foldl bind env) (cards.foldl applyCard entries) extra
    · simpa [bind_weights] using getWeights
    · have extended := (List.getElem?_append_left (l₂ := extra) (List.getElem?_eq_some_iff.mp found).1).trans found
      intro entry member
      exact TuplePool.View.frame _ _ (TupleDictionary.tuple_frame (heap ++ extra) root _ _ extended) _ _
        (valid_cards cards (heap ++ extra) entries (valid.append extra) views entry member)
    · exact distinct_cards cards entries distinct
    · apply positive_cards cards entries positive
      intro card member
      have allowed : card.data ∈ backData := by rw [← correspondence]; exact List.mem_map.mpr ⟨card, member, rfl⟩
      simp only [backData, List.mem_cons, List.not_mem_nil, or_false, Card.data, Prod.mk.injEq] at allowed
      omega
    · intro landing
      simpa [moveData, community, correspondence] using lookup_cards cards entries landing
  · simp only [community, decide_false]
    let pair := Mutable.Object.tuple [.scalar (.int destination), .scalar (.bool false)]
    let middle := heap ++ [pair]
    have allocated : HeapMemoCalls.Evaluates program pool.spec env
        (.tuple [.name "destination", .constant (.bool false)]) heap (.ref heap.length) middle globals :=
      HeapMemoCalls.Evaluates.tuple (.cons (HeapMemoCalls.Evaluates.lookup getDestination)
        (.cons HeapMemoCalls.Evaluates.constant .nil)) (TuplePool.preserves_append pool.catalog heap [pair])
    apply HeapMemoCalls.BlockReturnsRel.assign allocated.rel
    rintro value after ⟨rfl, same⟩
    subst after
    have extended : middle[root]? = some (.keyDict (encode entries)) :=
      (List.getElem?_append_left (l₂ := [pair]) (List.getElem?_eq_some_iff.mp found).1).trans found
    have key : TuplePool.View middle (.tuple (pool.pair destination false)) (.ref heap.length) :=
      (TuplePool.view_scalars middle [.int destination, .bool false] _).mpr ⟨heap.length, rfl, by simp [middle, pair]⟩
    let nextEnv := HeapCalls.setLocal env "outcome" (.ref heap.length)
    have nextWeights : nextEnv "weights" = some (.ref root) := by simpa [nextEnv, HeapCalls.setLocal] using getWeights
    apply increment_continue nextEnv middle globals root entries (destination, false) heap.length 16 (.constant (.int 16))
      extended nextWeights (by simp [nextEnv, HeapCalls.setLocal]) key HeapCalls.Evaluates.constant
    apply rest nextEnv (add entries (destination, false) heap.length 16) [pair] nextWeights
    · exact increment_valid middle root entries (destination, false) heap.length 16 extended (valid.append [pair]) key
    · exact distinct.added _ _ _
    · exact positive.added _ _ _ (by decide)
    · intro landing
      simpa [total, moveData, community] using lookup_add entries (destination, false) landing heap.length 16

end solvers.«84».resolve_landing.weights
