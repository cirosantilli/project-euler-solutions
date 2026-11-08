import solvers.«96.py.lean.d».Records

namespace solvers.«96».parse_puzzles
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

/-- The source parser collects every encoded board in order, including repeated records. -/
theorem loop_result {remaining : List String} {solutions : List Board} (records : Sudoku.Text.Records remaining solutions)
    (lines : List String) (position address : Nat) (base : Mutable.Heap) (boards : List Board)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (suffix : lines.drop position = remaining)
    (found : base[address]? = some (.list (lines.map HeapTextLines.textValue)))
    (locals : Locals address base.length position env) :
    ∃ parsed, List.Forall₂ Correct parsed solutions ∧
      BlockReturns program env [loop, finish] (work base boards) (.ref base.length) (work base (boards ++ parsed)) globals := by
  induction records generalizing position boards env with
  | nil =>
      have outside : ¬ position < lines.length := by
        have size := congrArg List.length suffix
        simp only [List.length_drop, List.length_nil] at size
        omega
      refine ⟨[], .nil, ?_⟩
      simp only [List.append_nil]
      apply BlockReturns.while_loop (flag := false)
        (measure lines position address (work base boards) env globals
          ((frame base boards address (List.getElem?_eq_some_iff.mp found).1).trans found) locals.lines locals.index)
        (by simp [Mutable.truth, scalarTruth, outside])
      exact BlockReturns.return_value (Evaluates.lookup locals.puzzles)
  | @skip line rest solutions ordinary tail ih =>
      have atPosition : lines[position]? = some line := by
        simpa [suffix] using (List.head?_drop (l := lines) (i := position)).symm
      have inside := (List.getElem?_eq_some_iff.mp atPosition).1
      have nextSuffix : lines.drop (position + 1) = rest := by simp [List.drop_add_one_eq_tail_drop, suffix]
      let ready := setLocal env "i" (number (position + 1))
      have nextLocals : Locals address base.length (position + 1) ready := by
        constructor
        · simpa [ready, setLocal] using locals.lines
        · simpa [ready, setLocal] using locals.puzzles
        · simp [ready, setLocal]
      obtain ⟨parsed, correct, continued⟩ := ih (position + 1) boards ready nextSuffix nextLocals
      have current := (frame base boards address (List.getElem?_eq_some_iff.mp found).1).trans found
      refine ⟨parsed, correct, ?_⟩
      apply iteration lines position address (work base boards) _ env globals _ current locals.lines locals.index inside
      apply BlockReturns.branch (flag := false)
        (read_header lines position address line (work base boards) env globals current locals.lines locals.index atPosition)
        (by simp [Mutable.truth, scalarTruth, ordinary])
      apply BlockReturns.augAssign (afterOperation := work base boards) (value := number (position + 1)) locals.index Evaluates.constant
      · simp [inplace, Mutable.binary, scalarBinary, number]
      exact continued
  | @puzzle header board solved rest solutions isHeader completed unique tail ih =>
      have atPosition : lines[position]? = some header := by
        simpa [suffix] using (List.head?_drop (l := lines) (i := position)).symm
      have inside := (List.getElem?_eq_some_iff.mp atPosition).1
      have afterHeader : lines.drop (position + 1) = Encoding.rows board ++ rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      have selected : (lines.drop (position + 1)).take 9 = Encoding.rows board := by
        rw [afterHeader]
        have size : (Encoding.rows board).length = 9 := by simp [Encoding.rows]
        simpa only [size] using (List.take_left (l₁ := Encoding.rows board) (l₂ := rest))
      have nextSuffix : lines.drop (position + 10) = rest := by
        have dropped := congrArg (List.drop 9) afterHeader
        simpa [List.drop_drop, Encoding.rows] using dropped
      let withGrid := setLocal env "grid" (.ref (work base boards).length)
      let ready := setLocal withGrid "i" (number (position + 10))
      have nextLocals : Locals address base.length (position + 10) ready := by
        constructor
        · simpa [ready, withGrid, setLocal] using locals.lines
        · simpa [ready, withGrid, setLocal] using locals.puzzles
        · simp [ready, setLocal]
      obtain ⟨parsed, correct, continued⟩ := ih (position + 10) (boards ++ [board]) ready nextSuffix nextLocals
      have current := (frame base boards address (List.getElem?_eq_some_iff.mp found).1).trans found
      refine ⟨board :: parsed, .cons ⟨completed, unique⟩ correct, ?_⟩
      apply iteration lines position address (work base boards) _ env globals _ current locals.lines locals.index inside
      apply BlockReturns.branch (flag := true)
        (read_header lines position address header (work base boards) env globals current locals.lines locals.index atPosition)
        (by simp [Mutable.truth, scalarTruth, isHeader])
      apply (collect_result lines position address board base boards env globals current locals selected).returns
      simpa only [List.append_assoc, List.singleton_append] using continued

end
end solvers.«96».parse_puzzles
