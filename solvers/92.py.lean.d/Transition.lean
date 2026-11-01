import solvers.«92.py.lean.d».Traversal
import lang.py.FormalSW.DictionaryCalls

namespace solvers.«92».transition
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := traversal.program
abbrev body : List Stmt :=
  [.assign [.name "following"] (.dict []), traversal.loop,
    .assign [.name "counts"] (.name "following")]

theorem source :
    (match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? with
      | some (Stmt.forIn _ _ prefixBody _) => prefixBody.drop 1
      | _ => []) = body := rfl

def rebound : List String := "counts" :: "following" :: traversal.rebound

def LocalsFrame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ rebound → after name = before name

/-- Each prefix digit builds a fresh distribution and retains every existing heap object. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (address ceiling : Nat) (entries : table.Entries)
    (counts : before "counts" = some (.ref address))
    (limit : before "ceiling" = some (table.number ceiling))
    (found : table.Found heap address entries) (valid : table.Keys heap entries) :
    ∃ result extra finalLocals,
      BlockContinues program before body heap finalLocals (heap ++ extra) globals ∧
      table.Found (heap ++ extra) heap.length result ∧ table.Keys (heap ++ extra) result ∧
      table.view result = Prefix.transition ceiling (table.view entries) ∧
      finalLocals "counts" = some (.ref heap.length) ∧ LocalsFrame before finalLocals := by
  let allocated := heap ++ [Mutable.Object.dict []]
  let initial := setLocal before "following" (.ref heap.length)
  have fresh : table.Found allocated heap.length [] := Or.inr ⟨rfl, by simp [allocated]⟩
  have different : address ≠ heap.length := Nat.ne_of_lt found.inside
  obtain ⟨result, finalHeap, locals, ran, finalFound, finalKeys, view, heapFrame, localFrame⟩ :=
    traversal.result allocated initial globals address heap.length ceiling entries []
      (by simpa [initial, setLocal] using counts) (by simp [initial, setLocal])
      (by simpa [initial, setLocal] using limit) (found.append _) (valid.append _)
      fresh (by intro entry member; cases member) different
  have retained : heap <+: finalHeap := by
    apply List.prefix_iff_getElem?.mpr
    intro index inside
    exact heapFrame index heap[index] (by omega)
      ((List.getElem?_append_left inside).trans (List.getElem?_eq_getElem inside))
  obtain ⟨extra, extension⟩ := retained
  have following : locals "following" = some (.ref heap.length) :=
    (traversal.local_following initial locals localFrame).trans (by simp [initial, setLocal])
  let finalLocals := setLocal locals "counts" (.ref heap.length)
  refine ⟨result, extra, finalLocals, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [extension]
    exact (BlockContinues.assign (Evaluates.dictionary .nil)).append
      (ran.append (BlockContinues.assign (Evaluates.lookup following)))
  · simpa only [extension] using finalFound
  · simpa only [extension] using finalKeys
  · simpa only [Prefix.transition, table.view, List.map_nil] using view
  · simp [finalLocals, setLocal]
  · intro name keep
    simp only [rebound, List.mem_cons, not_or] at keep
    simp only [finalLocals, setLocal, if_neg keep.1]
    rw [localFrame name keep.2.2]
    simp [initial, setLocal, keep.2.1]

/-- The represented distribution remains exact after reading one decimal digit. -/
theorem exact_result {bound ceiling : Nat} {before after : table.Entries}
    (represented : Prefix.Exact bound (table.view before)) (small : ceiling < 10)
    (changed : table.view after = Prefix.transition ceiling (table.view before)) :
    Prefix.Exact (10 * bound + ceiling) (table.view after) := by
  rw [changed]
  exact represented.transition ceiling small

end solvers.«92».transition
