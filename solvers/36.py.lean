import ProjectEulerStatements.Lib.DoublePalindromeSearch
import data.«project-euler-statements».data.py.project_euler_statements.«p36.py»
import «.formalsw».solvers.«36.py.ast»
import lang.py.FormalSW.IntegerParsing
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.BinaryFormatting
import ProjectEulerStatements.Lib.DecimalDigits
import ProjectEulerStatements.Lib.DoublePalindromeMirrors
import ProjectEulerStatements.Lib.DoublePalindromeBinary

set_option Elab.async false

namespace solvers.«36»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.DoublePalindromes
private abbrev program := Solvers.P36Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev decimal (n : Nat) : String := toString (n : Int)
private abbrev text (n : Nat) : Mutable.Value := .scalar (.string (decimal n))
private abbrev numbers (items : List Nat) : List Mutable.Value := items.map number

private theorem decimal_characters (n : Nat) (positive : 0 < n) :
    (decimal n).toList = (decimalDigits n).map Nat.digitChar := by
  simpa [decimal, toString, Int.repr, Nat.repr, decimalDigits] using
    ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n positive

private theorem odd_text (n : Nat) (positive : 0 < n) :
    decimal n ++ String.ofList (decimal n).toList.dropLast.reverse = decimal (oddPalindrome n) := by
  apply String.toList_injective
  rw [String.toList_append, String.toList_ofList, decimal_characters n positive,
    decimal_characters (oddPalindrome n) (oddPalindrome_positive n positive),
    decimalDigits_oddPalindrome n positive]
  simp [oddDigits, List.map_reverse, List.map_dropLast]

private theorem even_text (n : Nat) (positive : 0 < n) :
    decimal n ++ String.ofList (decimal n).toList.reverse = decimal (evenPalindrome n) := by
  apply String.toList_injective
  rw [String.toList_append, String.toList_ofList, decimal_characters n positive,
    decimal_characters (evenPalindrome n) (evenPalindrome_positive n positive),
    decimalDigits_evenPalindrome n positive]
  simp [evenDigits, List.map_reverse]

private theorem integer_evaluates (n : Nat) (expression : Expr)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (evaluated : Evaluates program env expression heap (text n) heap) :
    Evaluates program env (.builtinCall .int [expression]) heap (number n) heap := by
  apply Evaluates.builtin (.cons evaluated .nil)
  simp only [primitive, text, decimal, List.mapM_cons, List.mapM_nil, pure_bind]
  rw [Calls.integerConversion_int]
  rfl

private abbrev oddExpression : Expr := .builtinCall .int
  [.binOp (.name "text") .add (.subscript (.name "text")
    (.slice (some (.unaryOp .uSub (.constant (.int 2)))) none
      (some (.unaryOp .uSub (.constant (.int 1))))))]
private abbrev evenExpression : Expr := .builtinCall .int
  [.binOp (.name "text") .add (.subscript (.name "text")
    (.slice none none (some (.unaryOp .uSub (.constant (.int 1))))))]

private theorem even_evaluates (n : Nat) (positive : 0 < n)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getS : env "text" = some (text n)) :
    Evaluates program env evenExpression heap (number (evenPalindrome n)) heap := by
  apply integer_evaluates
  apply Evaluates.binary (av := text n)
    (bv := .scalar (.string (String.ofList (decimal n).toList.reverse)))
    (middle := heap) (afterRight := heap) (Evaluates.lookup getS)
  · apply Evaluates.slice (.cons (Evaluates.lookup getS) (.cons Evaluates.constant
      (.cons Evaluates.constant (.cons (Evaluates.unary Evaluates.constant rfl) .nil))))
    change (match Strings.slice (.string (decimal n)) .none .none (.int (-1)) with
      | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
      | .error error => some (Except.error error, heap)) = _
    rw [Strings.slice_reverse]
  · simpa only [even_text n positive] using
      (show Mutable.binary .add (text n) (.scalar (.string (String.ofList (decimal n).toList.reverse))) heap =
        some (.ok (.scalar (.string (decimal n ++ String.ofList (decimal n).toList.reverse))), heap) from rfl)

private theorem odd_evaluates (n : Nat) (positive : 0 < n)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getS : env "text" = some (text n)) :
    Evaluates program env oddExpression heap (number (oddPalindrome n)) heap := by
  apply integer_evaluates
  apply Evaluates.binary (av := text n)
    (bv := .scalar (.string (String.ofList (decimal n).toList.dropLast.reverse)))
    (middle := heap) (afterRight := heap) (Evaluates.lookup getS)
  · apply Evaluates.slice (.cons (Evaluates.lookup getS)
      (.cons (Evaluates.unary Evaluates.constant rfl) (.cons Evaluates.constant
        (.cons (Evaluates.unary Evaluates.constant rfl) .nil))))
    have sliced : Strings.slice (.string (decimal n)) (.int (-2)) .none (.int (-1)) =
        .ok (.string (String.ofList (decimal n).toList.dropLast.reverse)) := by
      change (match FormalSW.Py.Builtin.Sequence.slice (decimal n).toList
          { start := some (-2), step := some (-1) } with
        | .ok characters => Except.ok (Constant.string (String.ofList characters))
        | .error _ => Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)) = _
      have indices := FormalSW.Py.Builtin.Sequence.slice_reverse_skip (decimal n).toList 1
      rw [← List.dropLast_eq_take] at indices
      simp only [Nat.cast_one, show -(1 + 1 : Int) = -2 from rfl] at indices
      rw [indices]
    change (match Strings.slice (.string (decimal n)) (.int (-2)) .none (.int (-1)) with
      | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
      | .error error => some (Except.error error, heap)) = _
    rw [sliced]
  · simpa only [odd_text n positive] using
      (show Mutable.binary .add (text n) (.scalar (.string (String.ofList (decimal n).toList.dropLast.reverse))) heap =
        some (.ok (.scalar (.string (decimal n ++ String.ofList (decimal n).toList.dropLast.reverse))), heap) from rfl)

namespace is_palindrome_base2
private abbrev binaryDigits (n : Nat) := String.ofList (Nat.toDigits 2 n)

private theorem binary_suffix (n : Nat) (heap : Mutable.Heap) :
    Mutable.getSlice (.scalar (.string (Strings.binaryRepresentation n))) (.scalar (.int 2))
      (.scalar .none) (.scalar .none) heap =
      some (.ok (.scalar (.string (binaryDigits n))), heap) := by
  have sliced : Strings.slice (.string (Strings.binaryRepresentation n)) (.int 2) .none .none =
      .ok (.string (binaryDigits n)) := by
    change (match FormalSW.Py.Builtin.Sequence.slice (Strings.binaryRepresentation n).toList
        { start := some 2 } with
      | .ok characters => Except.ok (Constant.string (String.ofList characters))
      | .error _ => Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)) = _
    have suffix := FormalSW.Py.Builtin.Sequence.slice_suffix
      (Strings.binaryRepresentation n).toList 2
    norm_num only at suffix
    rw [suffix, Strings.binaryRepresentation_nat]
    rfl
  change (match Strings.slice (.string (Strings.binaryRepresentation n)) (.int 2) .none .none with
    | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
    | .error error => some (Except.error error, heap)) = _
  rw [sliced]

theorem result (n : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "is_palindrome_base2" [number n] heap
      (.scalar (.bool (ProjectEulerStatements.P36.isPalindromeBase 2 n))) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .scalar (.string (binaryDigits n))) (middle := heap)
  · apply Evaluates.slice (.cons
      (Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) rfl)
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
    exact binary_suffix n heap
  · apply BlockReturns.return_value
    apply Evaluates.compare (av := .scalar (.string (binaryDigits n)))
      (bv := .scalar (.string (String.ofList (binaryDigits n).toList.reverse)))
      (middle := heap) (afterRight := heap) (Evaluates.lookup rfl)
    · apply Evaluates.slice (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant
        (.cons Evaluates.constant (.cons (Evaluates.unary Evaluates.constant rfl) .nil))))
      change (match Strings.slice (.string (binaryDigits n)) .none .none (.int (-1)) with
        | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
        | .error error => some (Except.error error, heap)) = _
      rw [Strings.slice_reverse]
    · simp [HeapCalls.compare, scalarCompare, binaryDigits,
        String.ofList_inj, binary_string_palindrome]

end is_palindrome_base2

private abbrev generate := ProjectEulerStatements.DoublePalindromeSearch.generate
private abbrev scan := ProjectEulerStatements.DoublePalindromeSearch.scan

namespace generate_decimal_palindromes
private def collectorHeap (base : Mutable.Heap) (values : List Nat) : Mutable.Heap :=
  base ++ [.list (numbers values)]
private def selected (limit value : Nat) : List Nat := if value < limit then [value] else []
private theorem append_selected (limit value : Nat) (name : String)
    (base : Mutable.Heap) (before : List Nat) (env : HeapCalls.Env)
    (getValue : env name = some (number value)) (getLimit : env "limit" = some (number limit))
    (getPals : env "pals" = some (.ref base.length)) :
    BlockContinues program env [.ifThen (.compare (.name name) [(.lt, .name "limit")])
      [.expr (.call (.attribute (.name "pals") "append") [.name name])] []]
      (collectorHeap base before) env (collectorHeap base (before ++ selected limit value)) := by
  have tested : Evaluates program env (.compare (.name name) [(.lt, .name "limit")])
      (collectorHeap base before) (.scalar (.bool (decide (value < limit)))) (collectorHeap base before) :=
    Evaluates.compare (Evaluates.lookup getValue) (Evaluates.lookup getLimit)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases inside : value < limit
  · apply BlockContinues.branch (flag := true) (by simpa [inside] using tested) rfl
    apply BlockContinues.discard
    have appended := Evaluates.append (globals := []) (module := program) (env := env)
      (base := .name "pals") (expression := .name name) (value := number value)
      (heap := collectorHeap base before) (Evaluates.lookup getPals)
      (by simp [collectorHeap] : (collectorHeap base before)[base.length]? = some (.list (numbers before)))
      (Evaluates.lookup getValue)
      (by simp [collectorHeap] : (collectorHeap base before)[base.length]? = some (.list (numbers before)))
    simpa [collectorHeap, numbers, selected, inside, List.set_append_right] using appended
  · simpa only [selected, if_neg inside, List.append_nil] using
      (BlockContinues.branch (flag := false) (by simpa [inside] using tested) rfl BlockContinues.nil)


private def body : List Stmt :=
  [.assign [.name "text"] (.builtinCall .str [.name "prefix"]),
   .assign [.name "odd"] oddExpression,
   .ifThen (.compare (.name "odd") [(.gtE, .name "limit")]) [.return (.name "pals")] [],
   .expr (.call (.attribute (.name "pals") "append") [.name "odd"]),
   .assign [.name "even"] evenExpression,
   .ifThen (.compare (.name "even") [(.lt, .name "limit")])
     [.expr (.call (.attribute (.name "pals") "append") [.name "even"])] [],
   .augAssign (.name "prefix") .add (.constant (.int 1))]

private theorem loop (limit seed : Nat) (positive : 0 < seed) (base : Mutable.Heap) (before : List Nat)
    (env : HeapCalls.Env) (getLimit : env "limit" = some (number limit))
    (getPals : env "pals" = some (.ref base.length)) (getSeed : env "prefix" = some (number seed)) :
    BlockReturns program env [.whileDo (.constant (.bool true)) body []] (collectorHeap base before)
      (.ref base.length) (collectorHeap base (before ++ ProjectEulerStatements.DoublePalindromeSearch.generateFrom limit seed)) := by
  let e1 := setLocal env "text" (text seed)
  let e2 := setLocal e1 "odd" (number (oddPalindrome seed))
  have first : BlockContinues program env [body[0]'(by decide)] (collectorHeap base before) e1
      (collectorHeap base before) :=
    BlockContinues.assign (Evaluates.builtin (.cons (Evaluates.lookup getSeed) .nil) rfl)
  have second : BlockContinues program e1 [body[1]'(by decide)] (collectorHeap base before) e2
      (collectorHeap base before) := BlockContinues.assign (odd_evaluates seed positive e1 _ rfl)
  have tested : Evaluates program e2 (.compare (.name "odd") [(.gtE, .name "limit")])
      (collectorHeap base before) (.scalar (.bool (decide (limit ≤ oddPalindrome seed)))) (collectorHeap base before) :=
    Evaluates.compare (Evaluates.lookup rfl)
      (Evaluates.lookup (by simpa [e2, e1, setLocal] using getLimit))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  rw [ProjectEulerStatements.DoublePalindromeSearch.generateFrom_step limit seed positive]
  apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
  apply BlockContinues.returns (first.append second)
  by_cases inside : oddPalindrome seed < limit
  · rw [if_pos inside]
    apply BlockReturns.branch (flag := false) (by simpa [inside, Nat.not_le.mpr inside] using tested) rfl
    let afterOdd := before ++ [oddPalindrome seed]
    let e3 := setLocal e2 "even" (number (evenPalindrome seed))
    let e4 := setLocal e3 "prefix" (number (seed + 1))
    have appended : BlockContinues program e2 [body[3]'(by decide)] (collectorHeap base before)
        e2 (collectorHeap base afterOdd) := by
      apply BlockContinues.discard
      have appended := Evaluates.append (globals := []) (module := program) (env := e2) (base := .name "pals")
        (expression := .name "odd") (value := number (oddPalindrome seed))
        (heap := collectorHeap base before)
        (Evaluates.lookup (by simpa [e2, e1, setLocal] using getPals))
        (by simp [collectorHeap] : (collectorHeap base before)[base.length]? = some (.list (numbers before)))
        (Evaluates.lookup rfl)
        (by simp [collectorHeap] : (collectorHeap base before)[base.length]? = some (.list (numbers before)))
      simpa [collectorHeap, afterOdd, numbers, List.map_append, List.set_append_right] using appended
    have evenValue : BlockContinues program e2 [body[4]'(by decide)] (collectorHeap base afterOdd)
        e3 (collectorHeap base afterOdd) :=
      BlockContinues.assign (even_evaluates seed positive e2 _ (by simp [e2, e1, setLocal]))
    have evenAppended : BlockContinues program e3 [body[5]'(by decide)] (collectorHeap base afterOdd)
        e3 (collectorHeap base (afterOdd ++ selected limit (evenPalindrome seed))) :=
      append_selected limit (evenPalindrome seed) "even" base afterOdd e3 rfl
        (by simpa [e3, e2, e1, setLocal] using getLimit) (by simpa [e3, e2, e1, setLocal] using getPals)
    have incremented : BlockContinues program e3 [body[6]'(by decide)]
        (collectorHeap base (afterOdd ++ selected limit (evenPalindrome seed))) e4
        (collectorHeap base (afterOdd ++ selected limit (evenPalindrome seed))) := by
      apply BlockContinues.augAssign (left := number seed) (right := number 1)
        (by simpa [e3, e2, e1, setLocal] using getSeed) Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number]
    have rowEquals : afterOdd ++ selected limit (evenPalindrome seed) =
        before ++ ProjectEulerStatements.DoublePalindromeSearch.row limit seed := by
      by_cases evenInside : evenPalindrome seed < limit <;>
        simp [afterOdd, selected, ProjectEulerStatements.DoublePalindromeSearch.row, inside, evenInside, List.append_assoc]
    have continued := appended.append (evenValue.append (evenAppended.append incremented))
    rw [rowEquals] at continued
    have rest := loop limit (seed + 1) (by omega) base
      (before ++ ProjectEulerStatements.DoublePalindromeSearch.row limit seed) e4
      (by simpa [e4, e3, e2, e1, setLocal] using getLimit)
      (by simpa [e4, e3, e2, e1, setLocal] using getPals) rfl
    simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or,
      Bool.false_eq_true, if_false, List.nil_append, List.append_nil, List.append_assoc] using continued.returns rest
  · rw [if_neg inside, List.append_nil]
    apply BlockReturns.branch (flag := true) (by simpa [Nat.le_of_not_gt inside] using tested) rfl
    exact BlockReturns.return_value (Evaluates.lookup (by simpa [e2, e1, setLocal] using getPals))
termination_by limit - seed
decreasing_by
  have lower := ProjectEulerStatements.DoublePalindromeSearch.seed_le_odd seed positive
  omega

/-- The source stops exactly when later prefixes cannot contribute another value below the input. -/
theorem result (limit : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "generate_decimal_palindromes" [number limit] heap (.ref heap.length)
      (collectorHeap heap (generate limit)) := by
  let initial := functionEnv ((findFunctionIn program.body "generate_decimal_palindromes").getD ([], [])).1
    ((findFunctionIn program.body "generate_decimal_palindromes").getD ([], [])).2 [number limit] []
  let ready := setLocal (setLocal initial "pals" (.ref heap.length)) "prefix" (number 1)
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.list .nil)
  apply BlockReturns.assign Evaluates.constant
  simpa only [List.nil_append] using loop limit 1 (by decide) heap [] ready
    (by simp [ready, setLocal]; rfl) (by simp [ready, setLocal]) rfl

end generate_decimal_palindromes

namespace solve

private abbrev loopBody : List Stmt :=
  [.ifThen (.compare (.binOp (.name "n") .bitAnd (.constant (.int 1))) [(.eq, .constant (.int 0))])
      [.continueLoop] [],
    .ifThen (.call (.name "is_palindrome_base2") [.name "n"])
      [.augAssign (.name "total") .add (.name "n")] []]

private theorem loop (items remaining : List Nat) (position address total : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (numbers items)))
    (suffix : items.drop position = remaining)
    (getTotal : env "total" = some (number total)) :
    BlockReturns program env
      [.sequenceStepLoop "n" (.ref address) position loopBody [], .return (.name "total")]
      heap (number (remaining.foldl scanStep total)) heap := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (numbers items), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position total env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) sequence
      · have missing : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [numbers, missing]
      · exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | cons head tail ih =>
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let boundEnv := setLocal env "n" (number head)
      have getN : boundEnv "n" = some (number head) := by simp [boundEnv, setLocal]
      have boundTotal : boundEnv "total" = some (number total) := by
        simpa [boundEnv, setLocal] using getTotal
      have tailSuffix : items.drop (position + 1) = tail := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      have masked : Evaluates program boundEnv (.binOp (.name "n") .bitAnd (.constant (.int 1)))
          heap (number (head &&& 1)) heap := by
        apply Evaluates.binary (av := number head) (bv := number 1)
          (middle := heap) (afterRight := heap) (Evaluates.lookup getN) Evaluates.constant
        simp [Mutable.binary, scalarBitAnd, number, Nat.and_one_is_mod]
      have evenTest : Evaluates program boundEnv
          (.compare (.binOp (.name "n") .bitAnd (.constant (.int 1))) [(.eq, .constant (.int 0))])
          heap (.scalar (.bool (decide (head &&& 1 = 0)))) heap :=
        Evaluates.compare masked Evaluates.constant
          (by
            simp [HeapCalls.compare, scalarCompare, evalCmpOp, -Int.natCast_emod]
            omega)
      apply BlockReturns.sequence_step (source := .ref address) (value := number head) sequence (by simp [numbers, item])
      by_cases even : head &&& 1 = 0
      · apply BlockReturns.branch (flag := true) (by simpa only [even, decide_true] using evenTest) rfl
        apply BlockReturns.continue_loop (next := .sequenceStepLoop "n" (.ref address) (position + 1) loopBody [])
          (following := [.return (.name "total")]) rfl
        simpa only [List.foldl_cons, scanStep, if_pos even] using ih (position + 1) total boundEnv tailSuffix boundTotal
      · apply BlockReturns.branch (flag := false) (by simpa only [even, decide_false] using evenTest) rfl
        have palindrome : Evaluates program boundEnv (.call (.name "is_palindrome_base2") [.name "n"])
            heap (.scalar (.bool (ProjectEulerStatements.P36.isPalindromeBase 2 head))) heap :=
          Evaluates.call (.cons (Evaluates.lookup getN) .nil) (is_palindrome_base2.result head heap)
        cases accepted : ProjectEulerStatements.P36.isPalindromeBase 2 head with
        | false =>
            apply BlockReturns.branch (flag := false) (by simpa only [accepted] using palindrome) rfl
            apply BlockReturns.loop_resume
            simpa only [List.foldl_cons, scanStep, if_neg even, accepted, Bool.false_eq_true, if_false]
              using ih (position + 1) total boundEnv tailSuffix boundTotal
        | true =>
            apply BlockReturns.branch (flag := true) (by simpa only [accepted] using palindrome) rfl
            apply BlockReturns.augAssign (left := number total) (right := number head)
              (value := number (total + head)) (middle := heap) (afterOperation := heap)
              boundTotal (Evaluates.lookup getN)
            · simp [inplace, Mutable.binary, scalarBinary, number]
            · apply BlockReturns.loop_resume
              simpa only [List.foldl_cons, scanStep, if_neg even, accepted, ↓reduceIte]
                using ih (position + 1) (total + head) (setLocal boundEnv "total" (number (total + head)))
                  tailSuffix (by simp [setLocal])

/-- The executable seed scan has exactly the mathematical scan value for every limit. -/
theorem scan_result (limit : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [number limit] heap (number (scan limit))
      (heap ++ [.list (numbers (generate limit))]) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_sequence (items := numbers (generate limit))
    (by intros; simp)
    (Evaluates.call (.cons (Evaluates.lookup rfl) .nil)
      (generate_decimal_palindromes.result limit heap))
    (by simp [Mutable.sequence, generate_decimal_palindromes.collectorHeap])
  exact loop (generate limit) (generate limit) 0 heap.length 0 _ _ (by simp [generate_decimal_palindromes.collectorHeap]) rfl (by simp [setLocal])

theorem result (limit : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [number limit] heap (number (ProjectEulerStatements.P36.naive limit))
      (heap ++ [.list (numbers (generate limit))]) := by
  simpa only [ProjectEulerStatements.DoublePalindromeSearch.scan_correct limit] using scan_result limit heap

private abbrev inputEnv (limit : Int) : ScalarEnv :=
  fun name => if name = "limit" then some (.int limit) else none

private theorem condition_nonnegative (limit : Int)
    (condition : Calls.Condition program (inputEnv limit) Solvers.P36Ast.requires_solve_0) : 0 ≤ limit := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl

theorem «Returns» (limit : Int)
    (condition : Calls.Condition program (inputEnv limit) Solvers.P36Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  have nonnegative := condition_nonnegative limit condition
  lift limit to Nat using nonnegative
  exact (result limit []).scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int)
    (condition : Calls.Condition program (inputEnv limit) Solvers.P36Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P36Ast.module "naive" [.int limit]) := by
  have nonnegative := condition_nonnegative limit condition
  lift limit to Nat using nonnegative
  have optimized := (result limit []).scalar_finishes (values := [.int limit])
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p36.naive.result limit []
  have original := reference.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem «Supported» :
    (∀ limit : Int, Calls.Condition program (inputEnv limit) Solvers.P36Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, Calls.Condition program (inputEnv limit) Solvers.P36Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P36Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit condition
    exact («Returns» limit condition).supported
  · intro limit condition
    have nonnegative := condition_nonnegative limit condition
    lift limit to Nat using nonnegative
    obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p36.naive.result limit []
    exact (reference.scalar_returns (values := [.int limit])).supported

end solve
end solvers.«36»
