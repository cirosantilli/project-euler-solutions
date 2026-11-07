import solvers.«98.py.lean.d».Pattern
import ProjectEulerSolutions.Lib.KeyedReferences

namespace solvers.«98».word_patterns
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program number)
set_option Elab.async false

abbrev Rows := List (String × Nat)
abbrev work := KeyedReferences.work KeyedReferences.strings
abbrev text (word : String) : Mutable.Value := .scalar (.string word)
def tuple (word : String) : Mutable.Object := .tuple ((AnagramSquares.labels [] word.toList).map number)
def Valid (heap : Mutable.Heap) (rows : Rows) : Prop := ∀ row ∈ rows, heap[row.2]? = some (tuple row.1)

theorem Valid.append {heap : Mutable.Heap} {rows : Rows} (valid : Valid heap rows) (extra : Mutable.Heap) : Valid (heap ++ extra) rows :=
  fun row member => HeapCollection.prefix_found heap extra _ _ (valid row member)

abbrev body : List Stmt :=
  [.assign [.subscript (.name "word_pat") (.name "w")] (.call (.name "pattern_signature") [.name "w"])]

/-- Repeated words replace the dictionary value with another tuple of the same canonical pattern. -/
theorem insert (base : Mutable.Heap) (rows : Rows) (garbage : Mutable.Heap) (word : String) (env : HeapCalls.Env)
    (getWord : env "w" = some (text word)) (getDictionary : env "word_pat" = some (.ref base.length))
    (valid : Valid (work base rows garbage) rows) :
    ∃ extra address, BlockContinues program env body (work base rows garbage) env
      (work base (KeyedReferences.insert rows word address) (garbage ++ extra)) ∧
      Valid (work base (KeyedReferences.insert rows word address) (garbage ++ extra)) (KeyedReferences.insert rows word address) := by
  obtain ⟨after, address, returned, foundTuple, preserved⟩ := pattern_signature.result word (work base rows garbage)
  obtain ⟨extra, rfl⟩ := preserved
  refine ⟨extra, address, ?_, ?_⟩
  · apply BlockContinues.setItem (by intros; simp) (Evaluates.call (.cons (Evaluates.lookup getWord) .nil) returned)
      (.cons (Evaluates.lookup getDictionary) (.cons (Evaluates.lookup getWord) .nil))
    simpa only [KeyedReferences.append] using
      KeyedReferences.set_item KeyedReferences.strings base rows (garbage ++ extra) word address
  · intro row member
    have stored : (work base rows garbage ++ extra)[row.2]? = some (tuple row.1) := by
      rcases KeyedReferences.mem_insert rows word address row member with old | fresh
      · exact valid.append extra row old
      · simpa only [fresh] using foundTuple
    rw [KeyedReferences.append] at stored
    have different : base.length ≠ row.2 := by
      intro same
      rw [← same, KeyedReferences.found] at stored
      cases stored
    exact KeyedReferences.other_found KeyedReferences.strings base rows (KeyedReferences.insert rows word address)
      (garbage ++ extra) row.2 _ different stored

end solvers.«98».word_patterns
