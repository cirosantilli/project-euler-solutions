import solvers.«84.py.lean.d».LandingCardLoop
import ProjectEulerSolutions.Lib.TupleDictionaryRows

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

abbrev CardData := (Int × Bool) × Int
def Card.data (card : Card) : CardData := (card.entry.landing, card.entry.weight)
def cardKey (data : CardData) : HashKey := .tuple [.tuple (pool.pair data.1.1 data.1.2), .scalar (.int data.2)]

theorem card_of_view (heap : Mutable.Heap) (data : CardData) (value : Mutable.Value)
    (view : TuplePool.View heap (cardKey data) value) :
    ∃ card : Card, value = .ref card.row ∧ card.data = data ∧ card.View heap := by
  obtain ⟨row, returned, correspondence, found, key⟩ :=
    TupleDictionary.row_of_view heap (pool.pair data.1.1 data.1.2) (.int data.2) value view
  refine ⟨⟨row.address, ⟨data.1, row.keyAddress, data.2⟩⟩, returned, rfl, ?_, ?_⟩
  · simpa [correspondence.2] using found
  · simpa [correspondence.1] using key

theorem cards_of_views (heap : Mutable.Heap) (data : List CardData) (values : List Mutable.Value)
    (views : TuplePool.Views heap (data.map cardKey) values) :
    ∃ cards : List Card, values = cards.map (fun card => .ref card.row) ∧ cards.map Card.data = data ∧
      ∀ card ∈ cards, card.View heap := by
  induction data generalizing values with
  | nil =>
      cases values with
      | nil => exact ⟨[], rfl, rfl, by simp⟩
      | cons _ _ => exact False.elim views
  | cons datum data ih =>
      cases values with
      | nil => exact False.elim views
      | cons value values =>
          obtain ⟨card, rfl, same, view⟩ := card_of_view heap datum value views.1
          obtain ⟨cards, rfl, correspondence, rest⟩ := ih values views.2
          refine ⟨card :: cards, rfl, by simp [same, correspondence], ?_⟩
          intro current member
          rcases List.mem_cons.mp member with rfl | member
          · exact view
          · exact rest current member

def backData : List CardData := [((33, false), 14), ((0, false), 1), ((10, true), 1)]

def TablePost (base : Mutable.Heap) (data : List CardData) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ (source : Nat) (cards : List Card) (extra : Mutable.Heap), value = .ref source ∧ after = base ++ extra ∧ cards.map Card.data = data ∧
    after[source]? = some (.tuple (cards.map fun card => .ref card.row)) ∧ ∀ card ∈ cards, card.View after

/-- The compiler's pooled nested tuple exposes exactly the three Community Chest rows. -/
theorem back_table (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program pool.spec env pool.backThree heap (TablePost heap backData) globals := by
  apply (pool.back_result env heap globals).consequence
  rintro value after ⟨extra, same, root, items, rfl, found, views⟩
  obtain ⟨cards, rfl, correspondence, cardsView⟩ := cards_of_views after backData items views
  exact ⟨root, cards, extra, rfl, same, correspondence, found, cardsView⟩

end solvers.«84».resolve_landing.weights
