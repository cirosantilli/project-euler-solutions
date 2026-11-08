import solvers.«84.py.lean.d».LandingCardTable

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false
set_option maxRecDepth 2048

def backLoop : Stmt := .forIn (.tuple [.name "outcome", .name "count"]) pool.backThree [increment (.name "count")] []

theorem back_source : (pool.branch 2)[(1 : Nat)]?.bind (fun statement => match statement with
    | .forIn _ _ [.ifThen _ yes _] _ => yes[0]?
    | _ => none) = some backLoop := rfl

/-- Back three from Chance 36 executes the actual cached table and all three dictionary updates. -/
theorem back_continue (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root : Nat) (entries : List Entry)
    (found : heap[root]? = some (.keyDict (encode entries))) (getWeights : env "weights" = some (.ref root))
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : ∀ (cards : List Card) (extra : Mutable.Heap), cards.map Card.data = backData →
      (∀ card ∈ cards, card.View (heap ++ extra)) →
      HeapMemoCalls.BlockReturnsRel program pool.spec (cards.foldl bind env) body
        ((heap ++ extra).set root (.keyDict (encode (cards.foldl applyCard entries)))) post globals) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env (backLoop :: body) heap post globals := by
  intro anchors stack
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr pool.backThree env,
      .unpackSequence ["outcome", "count"] [increment (.name "count")] [] body env :: stack, heap⟩)
    (by trivial) rfl
  apply (back_table env heap globals anchors _).bind
  rintro final ⟨value, middle, rfl, source, cards, extra, rfl, rfl, correspondence, table, views⟩
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.block (.unpackStepLoop ["outcome", "count"] (.ref source) 0 [increment (.name "count")] [] :: body) env,
      stack, heap ++ extra⟩) (by trivial)
  · simp [HeapCalls.step, HeapCalls.run, Mutable.sequence, table, HeapCalls.sequenceSource]
  · have extended := (List.getElem?_append_left (l₂ := extra) (List.getElem?_eq_some_iff.mp found).1).trans found
    have run := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) pool.disjoint
      (card_loop cards [] env (heap ++ extra) globals source root entries table views extended getWeights body anchors stack)
      (TuplePool.preserves_frame pool.catalog _ _ (TupleDictionary.tuple_frame (heap ++ extra) root _ _ extended))
    apply run.bind
    rintro state rfl
    exact rest cards extra correspondence views anchors stack

def total (data : List CardData) (landing : Int × Bool) : Int :=
  (data.map fun item => if landing = item.1 then item.2 else 0).sum

theorem lookup_cards (cards : List Card) (entries : List Entry) (landing : Int × Bool) :
    (lookup (cards.foldl applyCard entries) landing).getD 0 =
      (lookup entries landing).getD 0 + total (cards.map Card.data) landing := by
  induction cards generalizing entries with
  | nil => simp [total]
  | cons card rest ih =>
      simp only [List.foldl_cons, ih, applyCard, lookup_add]
      simp [total, Card.data, Int.add_assoc]

theorem lookup_back (cards : List Card) (entries : List Entry) (landing : Int × Bool)
    (correspondence : cards.map Card.data = backData) :
    (lookup (cards.foldl applyCard entries) landing).getD 0 =
      (lookup entries landing).getD 0 +
        ((if landing = (33, false) then 14 else 0) +
         (if landing = (0, false) then 1 else 0) +
         (if landing = (10, true) then 1 else 0)) := by
  rw [lookup_cards, correspondence]
  simp [total, backData, Int.add_assoc]

end solvers.«84».resolve_landing.weights
