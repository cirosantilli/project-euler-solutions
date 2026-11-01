import solvers.«93.py.lean.d».MemoInvariant
import solvers.«93.py.lean.d».Accumulation
import ProjectEulerSolutions.Lib.RationalResultPreservation

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions
set_option Elab.async false

theorem MemoValid.collector_preserved {before after : Mutable.Heap} {memo address : Nat}
    {entries : List Memo.Entry} {collected : List Heap.SetEntry} (valid : MemoValid before memo entries)
    (collector : Heap.SetAt before address collected) (frame : PreservesExcept address before after)
    (uncached : address ∉ entries.map Memo.Entry.result) : MemoValid after memo entries := by
  constructor
  · have different : memo ≠ address := by
      intro equal
      have dictionary := valid.stored
      change (before[memo]?).bind Mutable.dictionaryEntries = some (Memo.encode entries) at dictionary
      rw [equal, collector.no_dictionary] at dictionary
      cases dictionary
    cases found : before[memo]? with
    | none =>
      have impossible := valid.stored
      simp [MemoAt, found] at impossible
    | some object =>
      have kept := frame.objects memo object different found
      simpa only [MemoAt, kept, found] using valid.stored
  · intro entry member
    apply (valid.results entry member).preserved_away collector frame
    intro equal
    exact uncached (List.mem_map.mpr ⟨entry, member, equal⟩)

/-- A completed child result is unioned into the private collector while every cache entry stays valid. -/
theorem update_valid (heap middle : Mutable.Heap) (address source memo : Nat)
    (collected : List Heap.SetEntry) (child : List Pair) (entries : List Memo.Entry)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (original : Heap.SetAt heap address collected) (getResult : env "results" = some (.ref address))
    (recursive : Evaluates program env recursiveResultExpr heap (.ref source) middle globals)
    (collector : Heap.SetAt middle address collected) (returned : Heap.ResultAt middle source child)
    (valid : MemoValid middle memo entries) (uncached : address ∉ entries.map Memo.Entry.result) :
    ∃ after final,
      BlockContinues program env [.expr updateExpr] heap env after globals ∧
      Heap.SetAt after address final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ collected.map Prod.fst ∨ Expected child pair) ∧
      MemoValid after memo entries ∧ PreservesExcept address middle after := by
  obtain ⟨incoming, input, inputMembers⟩ := returned
  obtain ⟨after, final, executed, found, members, unchanged, size⟩ :=
    update_result heap middle address source collected collected incoming env globals original getResult recursive collector input
  have frame : PreservesExcept address middle after := by
    refine ⟨by omega, ?_⟩
    intro other object different present
    rw [unchanged other different]
    exact present
  refine ⟨after, final, executed, found, ?_, valid.collector_preserved collector frame uncached, frame⟩
  intro pair
  rw [members, inputMembers]

end solvers.«93».all_results
