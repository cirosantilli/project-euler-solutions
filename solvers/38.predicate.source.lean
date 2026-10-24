import solvers.«38.concat.source»

namespace solvers.«38».is_pandigital
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.PandigitalDigits
set_option Elab.async false
private abbrev program := Solvers.P38Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev text (digits : List Nat) := String.ofList (digits.map Nat.digitChar)
private abbrev character (digit : Nat) : Mutable.Value := .scalar (.string (String.singleton (Nat.digitChar digit)))
private abbrev code : List Stmt := executableBody ((findFunctionIn program.body "is_pandigital").getD ([], [])).2
private abbrev loopBody : List Stmt :=
  match (code[2]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
variable {globals : Mutable.Globals}

private theorem digit_ordinal (digit : Nat) (decimal : digit < 10) :
    ((Nat.digitChar digit).toNat : Int) - 48 = (digit : Int) := by
  interval_cases digit <;> decide

private theorem ordinal_evaluates (digit : Nat) (decimal : digit < 10) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getChar : env "ch" = some (character digit)) :
    Evaluates (globals := globals) program env (.binOp (.builtinCall .ord [.name "ch"]) .sub (.constant (.int 48)))
      heap (number digit) heap := by
  apply Evaluates.binary (av := number (Nat.digitChar digit).toNat) (bv := number 48) (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getChar) .nil)
    change scalarResult (some (Strings.ord (.string (String.ofList [Nat.digitChar digit])))) heap = _
    rw [Strings.ord_character]
    rfl
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, number, digit_ordinal digit decimal]

/-- Digit range and duplicate checks return before the mask is extended. -/
private theorem scan_loop (width : Nat) (digits remaining : List Nat) (position mask : Nat)
    (suffix : digits.drop position = remaining) (decimal : ∀ digit ∈ remaining, digit < 10)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getMask : env "mask" = some (number mask))
    (getWidth : env "width" = some (number width)) :
    BlockReturns (globals := globals) program env
      (.sequenceStepLoop "ch" (.scalar (.string (text digits))) position loopBody [] :: code.drop 3)
      heap (.scalar (.bool (scan width remaining mask))) heap := by
  have sequence : Mutable.sequence (.scalar (.string (text digits))) heap = some (.ok (digits.map character), heap) := by
    simp [Mutable.sequence, text, character, List.map_map]
  induction remaining generalizing position mask env with
  | nil =>
      apply BlockReturns.sequence_done (source := .scalar (.string (text digits))) sequence
      · have missing : digits[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := digits) (i := position)).symm
        simp [missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons head tail ih =>
      have item : digits[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := digits) (i := position)).symm
      have headDecimal := decimal head (by simp)
      have tailDecimal : ∀ digit ∈ tail, digit < 10 := by
        intro digit member
        exact decimal digit (by simp [member])
      let charEnv := setLocal env "ch" (character head)
      let digitEnv := setLocal charEnv "digit" (number head)
      let bitEnv := setLocal digitEnv "bit" (number (2 ^ head))
      have getD : digitEnv "digit" = some (number head) := by simp [digitEnv, setLocal]
      have getBit : bitEnv "bit" = some (number (2 ^ head)) := by simp [bitEnv, setLocal]
      have oldMask : bitEnv "mask" = some (number mask) := by simpa [bitEnv, digitEnv, charEnv, setLocal] using getMask
      have widthHere : digitEnv "width" = some (number width) := by simpa [digitEnv, charEnv, setLocal] using getWidth
      apply BlockReturns.sequence_step (source := .scalar (.string (text digits))) (value := character head) sequence (by simp [item])
      apply BlockReturns.assign (ordinal_evaluates head headDecimal heap charEnv (by simp [charEnv, setLocal]))
      by_cases zero : head = 0
      · simp only [scan, if_pos zero]
        apply BlockReturns.branch (flag := true)
          (Evaluates.compare (av := number head) (bv := number 1) (value := .scalar (.bool true)) (finalHeap := heap)
            (Evaluates.lookup getD) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])) rfl
        exact BlockReturns.return_value Evaluates.constant
      · simp only [scan, if_neg zero]
        apply BlockReturns.branch (flag := false)
          (Evaluates.compare (av := number head) (bv := number 1) (value := .scalar (.bool false)) (finalHeap := heap)
            (Evaluates.lookup getD) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
        have outsideTest : Evaluates (globals := globals) program digitEnv
            (.compare (.name "digit") [(.gt, .name "width")]) heap
            (.scalar (.bool (decide (width < head)))) heap := by
          apply Evaluates.compare (av := number head) (bv := number width) (middle := heap) (afterRight := heap)
            (Evaluates.lookup getD) (Evaluates.lookup widthHere)
          simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        by_cases outside : width < head
        · rw [if_pos outside]
          apply BlockReturns.branch (flag := true) (by simpa [outside] using outsideTest) rfl
          exact BlockReturns.return_value Evaluates.constant
        · rw [if_neg outside]
          apply BlockReturns.branch (flag := false) (by simpa [outside] using outsideTest) rfl
          apply BlockReturns.assign (value := number (2 ^ head)) (middle := heap)
          · apply Evaluates.binary (av := number 1) (bv := number head) (middle := heap) (afterRight := heap)
              Evaluates.constant (Evaluates.lookup getD)
            simp [Mutable.binary, scalarShiftLeft, integerShiftLeft, number, show ¬ (head : Int) < 0 by omega]
          · have masked : Evaluates (globals := globals) program bitEnv (.binOp (.name "mask") .bitAnd (.name "bit"))
                heap (number (mask &&& 2 ^ head)) heap :=
              Evaluates.binary (Evaluates.lookup oldMask) (Evaluates.lookup getBit)
                (by simp [Mutable.binary, scalarBitAnd, number]; rfl)
            by_cases overlap : mask &&& 2 ^ head ≠ 0
            · rw [if_pos overlap]
              apply BlockReturns.branch (flag := true) masked (by simp [Mutable.truth, scalarTruth, overlap])
              exact BlockReturns.return_value Evaluates.constant
            · rw [if_neg overlap]
              have empty : mask &&& 2 ^ head = 0 := by omega
              apply BlockReturns.branch (flag := false) masked (by simp [Mutable.truth, scalarTruth, empty])
              apply BlockReturns.augAssign (left := number mask) (right := number (2 ^ head))
                (value := number (mask ||| 2 ^ head)) (middle := heap) (afterOperation := heap) oldMask (Evaluates.lookup getBit)
              · simp [inplace, Mutable.binary, scalarBitOr, number]
                rfl
              · have rest := ih (position + 1) (mask ||| 2 ^ head) (by simp [List.drop_add_one_eq_tail_drop, suffix])
                  tailDecimal (setLocal bitEnv "mask" (number (mask ||| 2 ^ head))) (by simp [setLocal])
                  (by simpa [setLocal, bitEnv, digitEnv, charEnv] using getWidth)
                simpa [loopBody, code, program, Solvers.P38Ast.module, findFunctionIn, executableBody,
                  Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, bitEnv, digitEnv, charEnv] using rest

/-- On decimal strings, the mask implementation is the variable-width permutation predicate. -/
theorem result (width : Nat) (digits : List Nat) (decimal : ∀ digit ∈ digits, digit < 10)
    (heap : Mutable.Heap) :
    ReturnsValue (globals := globals) program "is_pandigital" [.scalar (.string (text digits)), number width] heap
      (.scalar (.bool (check width digits))) heap := by
  let initial := functionEnv ((findFunctionIn program.body "is_pandigital").getD ([], [])).1
    ((findFunctionIn program.body "is_pandigital").getD ([], [])).2 [.scalar (.string (text digits)), number width] globals
  have getInput : initial "s" = some (.scalar (.string (text digits))) := by cases globals <;> rfl
  have getWidth : initial "width" = some (number width) := by cases globals <;> rfl
  have lengthTest : Evaluates (globals := globals) program initial
      (.compare (.builtinCall .len [.name "s"]) [(.notEq, .name "width")]) heap
      (.scalar (.bool (decide (digits.length ≠ width)))) heap := by
    apply Evaluates.compare (av := number digits.length) (bv := number width) (middle := heap) (afterRight := heap)
      (Evaluates.builtin (.cons (Evaluates.lookup getInput) .nil) (by simp [primitive, Mutable.length, text])) (Evaluates.lookup getWidth)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  apply BlockReturns.function rfl rfl
  by_cases same : digits.length = width
  · simp only [check, same, decide_true, Bool.true_and]
    apply BlockReturns.branch (flag := false) (by simpa only [same, ne_eq, not_true_eq_false, decide_false] using lengthTest) rfl
    apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.for_sequence (items := digits.map character) (by intros; simp)
      (Evaluates.lookup (by simpa [setLocal] using getInput)) (by simp [Mutable.sequence, text, character, List.map_map])
    exact scan_loop width digits digits 0 0 (by simp) decimal heap _ (by simp [setLocal]) (by simpa [setLocal] using getWidth)
  · simp only [check, same, decide_false, Bool.false_and]
    apply BlockReturns.branch (flag := true) (by simpa only [same, ne_eq, not_false_eq_true, decide_true] using lengthTest) rfl
    exact BlockReturns.return_value Evaluates.constant

end solvers.«38».is_pandigital
