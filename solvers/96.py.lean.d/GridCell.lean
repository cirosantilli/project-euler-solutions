import solvers.«96.py.lean.d».Projection
import ProjectEulerSolutions.Lib.SudokuEncoding
import lang.py.FormalSW.IntegerParsing

namespace solvers.«96».grid_to_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev text (s : String) : Mutable.Value := .scalar (.string s)
abbrev work (base : Mutable.Heap) (board : Board) (cursor : Nat) : Mutable.Heap :=
  base ++ [.list [number 511], .list ((Encoding.prepared board cursor).map fun n : Nat => number n)]

abbrev offset : Expr := .binOp (.binOp (.name "r") .mult (.constant (.int 9))) .add (.name "c")
abbrev fill : List Stmt :=
  [.assign [.name "d"] (.builtinCall .int [.name "ch"]),
   .assign [.subscript (.name "cands") offset]
     (.binOp (.constant (.int 1)) .leftShift (.binOp (.name "d") .sub (.constant (.int 1))))]
abbrev columnBody : List Stmt :=
  [.assign [.name "ch"] (.subscript (.name "row") (.name "c")),
   .ifThen (.compare (.name "ch") [(.notEq, .constant (.string "0"))]) fill []]
abbrev rowBody : List Stmt :=
  [.assign [.name "row"] (.subscript (.name "grid_lines") (.name "r")),
   .forIn (.name "c") (.call (.name "range") [.constant (.int 9)]) columnBody []]
abbrev definition := (findFunctionIn program.body "grid_to_cands").getD ([], [])

theorem source : definition.2 =
    [.assign [.name "cands"] (.binOp (.list [.name "ALL"]) .mult (.constant (.int 81))),
     .forIn (.name "r") (.call (.name "range") [.constant (.int 9)]) rowBody [],
     .return (.name "cands")] := rfl

theorem found (base : Mutable.Heap) (board : Board) (cursor : Nat) :
    (work base board cursor)[base.length + 1]? =
      some (.list ((Encoding.prepared board cursor).map fun n : Nat => number n)) := by simp [work]

theorem read_character (board : Board) (r c : Fin 9) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getRow : env "row" = some (text (Encoding.row board r)))
    (getColumn : env "c" = some (number c.val)) :
    Evaluates program env (.subscript (.name "row") (.name "c")) heap
      (text (String.singleton (Encoding.character (board (r, c))))) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getRow) (Evaluates.lookup getColumn)
  have inside : (c.val : Int) < 9 := by exact_mod_cast c.isLt
  have nonnegative : ¬ (c.val : Int) < 0 := by omega
  obtain ⟨_, selected⟩ := List.getElem_of_getElem? (Encoding.row_get board r c)
  simp [Mutable.getItem, Strings.get, scalarNumber, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, selected, not_le.mpr inside, nonnegative, text]

theorem store (base : Mutable.Heap) (board : Board) (r c : Fin 9) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (filled : board (r, c) = some digit)
    (getRow : env "r" = some (number r.val)) (getColumn : env "c" = some (number c.val))
    (getCandidates : env "cands" = some (.ref (base.length + 1))) (getDigit : env "d" = some (number (digit.val + 1))) :
    BlockContinues program env (fill.drop 1) (work base board (r.val * 9 + c.val)) env
      (work base board (r.val * 9 + c.val + 1)) globals := by
  let slot := Encoding.index (r, c)
  have shifted : Evaluates program env
      (.binOp (.constant (.int 1)) .leftShift (.binOp (.name "d") .sub (.constant (.int 1))))
      (work base board slot.val) (number (2 ^ digit.val)) (work base board slot.val) globals := by
    apply Evaluates.binary (middle := work base board slot.val) (afterRight := work base board slot.val) (av := number 1) (bv := number digit.val) Evaluates.constant
    · apply Evaluates.binary (middle := work base board slot.val) (afterRight := work base board slot.val) (av := number (digit.val + 1)) (bv := number 1)
        (Evaluates.lookup getDigit) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number]
    · have nonnegative : ¬ (digit.val : Int) < 0 := by omega
      simp [Mutable.binary, scalarShiftLeft, scalarNumber, integerShiftLeft, number, nonnegative]
  have position : Evaluates program env offset (work base board slot.val) (number slot.val)
      (work base board slot.val) globals := by
    apply Evaluates.binary (middle := work base board slot.val) (afterRight := work base board slot.val) (av := number (r.val * 9)) (bv := number c.val)
    · apply Evaluates.binary (middle := work base board slot.val) (afterRight := work base board slot.val) (Evaluates.lookup getRow) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number]
    · exact Evaluates.lookup getColumn
    · simp [Mutable.binary, scalarBinary, number, slot, Encoding.index]
  have updated : ((Encoding.prepared board slot.val).map fun n : Nat => number n).set slot.val (number (2 ^ digit.val)) =
      (Encoding.prepared board (slot.val + 1)).map fun n : Nat => number n := by
    have step := congrArg (List.map fun n => number (n : Nat)) (Encoding.prepared_step board slot)
    simpa only [List.map_set, slot, Encoding.cell_index, Encoding.mask_some board (r, c) digit filled] using step
  apply BlockContinues.setItem (by intros; simp) shifted (.cons (Evaluates.lookup getCandidates) (.cons position .nil))
  have nonnegative : ¬ (slot.val : Int) < 0 := by omega
  change Mutable.setItem (.ref (base.length + 1)) (number slot.val) (number (2 ^ digit.val))
    (work base board slot.val) = some (.ok (.scalar .none), work base board (slot.val + 1))
  simp [Mutable.setItem, Mutable.index, not_le.mpr slot.isLt, nonnegative, updated, work,
    List.set_append_right, Except.map]

theorem column (base : Mutable.Heap) (board : Board) (r c : Fin 9) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (getR : env "r" = some (number r.val))
    (getRow : env "row" = some (text (Encoding.row board r)))
    (getColumn : env "c" = some (number c.val)) (getCandidates : env "cands" = some (.ref (base.length + 1))) :
    ∃ finalEnv, BlockContinues program env columnBody (work base board (r.val * 9 + c.val))
      finalEnv (work base board (r.val * 9 + c.val + 1)) globals ∧
      (∀ name, name ∉ ["ch", "d"] → finalEnv name = env name) := by
  let ready := setLocal env "ch" (text (String.singleton (Encoding.character (board (r, c)))))
  have assigned := BlockContinues.assign (name := "ch")
    (read_character board r c env (work base board (r.val * 9 + c.val)) globals getRow getColumn)
  cases value : board (r, c) with
  | none =>
      have unchanged := Encoding.prepared_skip board (Encoding.index (r, c)) (by simpa using value)
      refine ⟨ready, ?_, ?_⟩
      · apply assigned.append
        simp only [work, show r.val * 9 + c.val + 1 = (Encoding.index (r, c)).val + 1 from rfl, ← unchanged]
        apply BlockContinues.branch (middle := work base board (r.val * 9 + c.val)) (flag := false) (value := .scalar (.bool false))
        · apply Evaluates.compare (middle := work base board (r.val * 9 + c.val)) (afterRight := work base board (r.val * 9 + c.val)) (Evaluates.lookup (show ready "ch" = some (text "0") by simp [ready, value, Encoding.character, setLocal])) Evaluates.constant
          rfl
        · rfl
        · exact .nil
      · intro name absent
        simp only [List.mem_cons, not_or] at absent
        simp [ready, setLocal, absent.1]
  | some digit =>
      let finalEnv := setLocal ready "d" (number (digit.val + 1))
      have parsed : Evaluates program ready (.builtinCall .int [.name "ch"])
          (work base board (r.val * 9 + c.val)) (number (digit.val + 1)) (work base board (r.val * 9 + c.val)) globals := by
        apply Evaluates.builtin (.cons (Evaluates.lookup (show ready "ch" = some (text (digit.val + 1).repr) by
          simp [ready, setLocal, value, Encoding.digit_text])) .nil)
        have parsedText : FormalSW.Py.Builtin.Int.parseText (digit.val + 1).repr = .ok ((digit.val + 1 : Nat) : Int) :=
          FormalSW.Py.Builtin.Int.parseText_nat (digit.val + 1)
        simp [primitive, Calls.integerConversion, parsedText, number, text]
        rfl
      refine ⟨finalEnv, ?_, ?_⟩
      · apply assigned.append
        apply BlockContinues.branch (middle := work base board (r.val * 9 + c.val)) (flag := true) (value := .scalar (.bool true))
        · apply Evaluates.compare (middle := work base board (r.val * 9 + c.val)) (afterRight := work base board (r.val * 9 + c.val)) (Evaluates.lookup (show ready "ch" = some (text (String.singleton (Encoding.character (some digit)))) by
            simp [ready, value, setLocal])) Evaluates.constant
          have nonzero : String.singleton (Encoding.character (some digit)) ≠ "0" := by
            exact (show ∀ digit : Digit, String.singleton (Encoding.character (some digit)) ≠ "0" from by decide) digit
          simp [HeapCalls.compare, scalarCompare, nonzero]
        · rfl
        · exact (BlockContinues.assign parsed).append (store base board r c digit finalEnv globals value
            (by simpa [finalEnv, ready, setLocal] using getR)
            (by simpa [finalEnv, ready, setLocal] using getColumn)
            (by simpa [finalEnv, ready, setLocal] using getCandidates) (by simp [finalEnv, setLocal]))
      · intro name absent
        simp only [List.mem_cons, not_or] at absent
        simp [finalEnv, ready, setLocal, absent.1, absent.2]

end
end solvers.«96».grid_to_cands
