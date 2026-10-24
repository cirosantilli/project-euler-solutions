import «.formalsw».solvers.«33.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.SetCalls
import ProjectEulerStatements.Lib.DigitCancellingCertificate
import ProjectEulerSolutions.Lib.DigitCancellingStrings
import data.«project-euler-statements».data.py.project_euler_statements.«p33.py»
import Mathlib.Data.List.Forall2

namespace solvers.«33»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.DigitCancellingFractions
open ProjectEulerSolutions.DigitCancellingStrings

private abbrev program := Solvers.P33Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev text (n : Nat) : Mutable.Value := .scalar (.string (decimal n))

set_option Elab.async false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

private theorem remainder_lt (n digit : Nat) (upper : n < 100) : remainder n digit < 10 := by
  unfold remainder
  split
  · omega
  · omega

private theorem decimal_evaluates (n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (expression : Expr)
    (evaluated : Evaluates program env expression heap (number n) heap globals) :
    Evaluates program env (.builtinCall .str [expression]) heap (text n) heap globals := by
  exact Evaluates.builtin (.cons evaluated .nil) rfl

private theorem replacement_evaluates (n digit : Nat) (lower : 10 ≤ n) (upper : n < 100)
    (positive : 0 < digit) (single : digit < 10) (present : hasDigit n digit)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (receiverName : String) (receiver : env receiverName = some (text n))
    (getDigit : env "c" = some (text digit)) :
    Evaluates program env
      (.call (.attribute (.name receiverName) "replace")
        [.name "c", .constant (.string ""), .constant (.int 1)])
      heap (text (remainder n digit)) heap globals := by
  apply Evaluates.replace (Evaluates.lookup receiver)
    (.cons (Evaluates.lookup getDigit) (.cons Evaluates.constant (.cons Evaluates.constant .nil)))
  · change some ((Except.ok (ε := ExceptionObject)) (Mutable.Value.scalar (.string
      (Strings.replace (decimal n) (decimal digit) "" 1))), heap) =
      some (.ok (text (remainder n digit)), heap)
    rw [(decimal_bridge n digit lower upper positive single).2 present]
  · intro name equal
    cases equal
    simp [receiver]

private theorem digit_integer_evaluates (n : Nat) (single : n < 10)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (expression : Expr)
    (evaluated : Evaluates program env expression heap (text n) heap globals) :
    Evaluates program env (.builtinCall .int [expression]) heap (number n) heap globals := by
  apply Evaluates.builtin (.cons evaluated .nil)
  simp only [primitive, text, List.mapM_cons, List.mapM_nil, pure_bind]
  rw [(one_digit_facts ⟨n, single⟩).2.1]
  rfl


/- Search heap representation. The result list has one stable address; temporary
unpacking tuples and appended fraction tuples retain their real heap addresses. -/
def PairValues (heap : Mutable.Heap) (values : List Mutable.Value)
    (pairs : List (Nat × Nat)) : Prop :=
  List.Forall₂ (fun value pair => ∃ address,
    value = .ref address ∧ heap[address]? = some (.tuple [number pair.1, number pair.2])) values pairs

private theorem PairValues.preserved {heap finalHeap : Mutable.Heap}
    {values : List Mutable.Value} {pairs : List (Nat × Nat)} (represented : PairValues heap values pairs)
    (unchanged : ∀ (address : Nat) (pair : Nat × Nat), heap[address]? = some (.tuple [number pair.1, number pair.2]) →
      finalHeap[address]? = some (.tuple [number pair.1, number pair.2])) :
    PairValues finalHeap values pairs := by
  induction represented with
  | nil => exact .nil
  | cons head tail ih =>
      rcases head with ⟨address, equal, found⟩
      exact .cons ⟨address, equal, unchanged address _ found⟩ ih

private theorem PairValues.append_heap (represented : PairValues heap values pairs)
    (extra : Mutable.Heap) : PairValues (heap ++ extra) values pairs := by
  apply PairValues.preserved represented
  intro address pair found
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private def workHeap (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap) :
    Mutable.Heap := base ++ (.list values :: garbage)

private theorem workHeap_found (base : Mutable.Heap) (values : List Mutable.Value)
    (garbage : Mutable.Heap) :
    (workHeap base values garbage)[base.length]? = some (.list values) := by simp [workHeap]

private theorem workHeap_append (base : Mutable.Heap) (values : List Mutable.Value)
    (garbage extra : Mutable.Heap) :
    workHeap base values garbage ++ extra = workHeap base values (garbage ++ extra) := by
  simp [workHeap, List.append_assoc]

private theorem workHeap_set (base : Mutable.Heap) (values replacement : List Mutable.Value)
    (garbage : Mutable.Heap) :
    (workHeap base values garbage).set base.length (.list replacement) =
      workHeap base replacement garbage := by
  simp [workHeap, List.set_append_right]

private theorem PairValues.replace_list
    (represented : PairValues (workHeap base values garbage) values pairs)
    (replacement : List Mutable.Value) :
    PairValues (workHeap base replacement garbage) values pairs := by
  apply PairValues.preserved represented
  intro address pair found
  have different : address ≠ base.length := by
    intro same
    subst address
    simp [workHeap] at found
  rw [← workHeap_set (values := values)]
  rw [List.getElem?_set_ne (Ne.symm different)]
  exact found

private abbrev findCode : List Stmt :=
  executableBody ((findFunctionIn program.body "find_curious_fractions").getD ([], [])).2
private abbrev numeratorBody : List Stmt :=
  match (findCode[1]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private abbrev denominatorBody : List Stmt :=
  match (numeratorBody[0]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private abbrev digitBody : List Stmt :=
  match (denominatorBody[2]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private abbrev cancellationBody : List Stmt :=
  match (digitBody[0]? : Option Stmt) with | some (.ifThen _ yes _) => yes | _ => []

/- A source loop body either advances to the next iteration or executes break.
The explicit boundary is the interpreter's ordinary continue/break semantics. -/
private def LoopIteration (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (broke : Bool) (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap)
    (globals : Mutable.Globals) : Prop :=
  ∀ next following anchors stack,
    Reaches program anchors
      ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
      ⟨.block (if broke then following else next :: following) finalEnv, stack, finalHeap⟩ globals

private theorem LoopIteration.done (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) : LoopIteration env [] heap false env heap globals := by
  intro next following anchors stack
  exact .next rfl .refl

private theorem LoopIteration.prepend
    (first : BlockContinues program env body heap middleEnv middleHeap globals)
    (rest : LoopIteration middleEnv tail middleHeap broke finalEnv finalHeap globals) :
    LoopIteration env (body ++ tail) heap broke finalEnv finalHeap globals := by
  intro next following anchors stack
  have tailRun : Reaches program anchors
      ⟨.block (tail ++ (.loopResume next :: following)) middleEnv, stack, middleHeap⟩
      ⟨.block (if broke then following else next :: following) finalEnv, stack, finalHeap⟩ globals := by
    simpa only [List.append_assoc, List.singleton_append] using rest next following anchors stack
  simpa only [List.append_assoc, List.singleton_append] using
    (first (tail ++ (.loopResume next :: following)) anchors stack).trans tailRun

private theorem LoopIteration.branch
    (evaluated : Evaluates program env condition heap value middle globals)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : LoopIteration env ((if flag then yes else no) ++ body) middle
      broke finalEnv finalHeap globals) :
    LoopIteration env (.ifThen condition yes no :: body) heap broke finalEnv finalHeap globals := by
  intro next following anchors stack
  apply Reaches.next rfl
  apply (evaluated anchors _).trans
  apply Reaches.next (following :=
    ⟨.block (((if flag then yes else no) ++ body) ++ [.loopResume next] ++ following) env,
      stack, middle⟩)
    (by simp [HeapCalls.step, run, truth, List.append_assoc])
  exact rest next following anchors stack

private theorem LoopIteration.skip
    (target : ∀ next following,
      Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    LoopIteration env (.continueLoop :: body) heap false env heap globals := by
  intro next following anchors stack
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact .next (by simp [HeapCalls.step, found]) .refl

private theorem LoopIteration.stop
    (target : ∀ next following,
      Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    LoopIteration env (.breakLoop :: body) heap true env heap globals := by
  intro next following anchors stack
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact .next (by simp [HeapCalls.step, found]) .refl

private structure SearchLocals (env : HeapCalls.Env) (address n d : Nat) : Prop where
  result : env "res" = some (.ref address)
  numerator : env "n" = some (number n)
  denominator : env "d" = some (number d)
  numeratorText : env "ns" = some (text n)
  denominatorText : env "ds" = some (text d)

private theorem SearchLocals.other (locals : SearchLocals env address n d)
    (name : String) (value : Mutable.Value)
    (unrelated : name ∉ ["res", "n", "d", "ns", "ds"]) :
    SearchLocals (setLocal env name value) address n d := by
  rcases locals with ⟨result, numerator, denominator, numeratorText, denominatorText⟩
  simp only [List.mem_cons, not_or] at unrelated
  constructor <;> simp_all [setLocal, eq_comm]

private theorem boolean_and (left right : Expr) (a b : Bool)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (first : Evaluates program env left heap (.scalar (.bool a)) heap globals)
    (second : Evaluates program env right heap (.scalar (.bool b)) heap globals) :
    Evaluates program env (.boolOp .and [left, right]) heap (.scalar (.bool (a && b))) heap globals := by
  cases a
  · exact Evaluates.boolean_stop first rfl rfl
  · exact Evaluates.boolean_next first rfl rfl (Evaluates.boolean_last second rfl)

private theorem boolean_or (left right : Expr) (a b : Bool)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (first : Evaluates program env left heap (.scalar (.bool a)) heap globals)
    (second : Evaluates program env right heap (.scalar (.bool b)) heap globals) :
    Evaluates program env (.boolOp .or [left, right]) heap (.scalar (.bool (a || b))) heap globals := by
  cases a
  · exact Evaluates.boolean_next first rfl rfl (Evaluates.boolean_last second rfl)
  · exact Evaluates.boolean_stop first rfl rfl

private theorem membership_evaluates (n digit : Nat) (lower : 10 ≤ n) (upper : n < 100)
    (positive : 0 < digit) (single : digit < 10)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (name : String) (receiver : env name = some (text n))
    (getDigit : env "c" = some (text digit)) :
    Evaluates program env (.membership (.name "c") (.name name) false)
      heap (.scalar (.bool (decide (hasDigit n digit)))) heap globals := by
  apply Evaluates.membership (.cons (Evaluates.lookup getDigit) (.cons (Evaluates.lookup receiver) .nil))
  rw [HeapCalls.membership_string, Strings.contains_eq_isInfix,
    (decimal_bridge n digit lower upper positive single).1]
  rfl

private theorem append_fraction (base : Mutable.Heap) (values : List Mutable.Value)
    (garbage : Mutable.Heap) (pairs : List (Nat × Nat)) (n d : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (getResult : env "res" = some (.ref base.length))
    (getN : env "n" = some (number n)) (getD : env "d" = some (number d)) :
    let address := (workHeap base values garbage).length
    let finalValues := values ++ [.ref address]
    let finalGarbage := garbage ++ [.tuple [number n, number d]]
    BlockContinues program env
      [.expr (.call (.attribute (.name "res") "append") [.tuple [.name "n", .name "d"]])]
      (workHeap base values garbage) env (workHeap base finalValues finalGarbage) globals ∧
    PairValues (workHeap base finalValues finalGarbage) finalValues (pairs ++ [(n, d)]) := by
  dsimp only
  let address := (workHeap base values garbage).length
  let extra : Mutable.Heap := [.tuple [number n, number d]]
  have extended := represented.append_heap extra
  rw [workHeap_append] at extended
  have old := extended.replace_list (values ++ [.ref address])
  have oldAddress : base.length < address := by
    simp only [address, workHeap, List.length_append, List.length_cons]
    omega
  have found : (workHeap base (values ++ [.ref address]) (garbage ++ extra))[address]? =
      some (.tuple [number n, number d]) := by
    rw [← workHeap_set (values := values), ← workHeap_append]
    rw [List.getElem?_set_ne (by omega : base.length ≠ address)]
    simp [extra, address]
  constructor
  · apply BlockContinues.discard
    have appended := Evaluates.append (module := program) (globals := globals) (current := values)
      (Evaluates.lookup getResult)
      (workHeap_found base values garbage)
      (Evaluates.tuple (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getD) .nil)) (by simp))
      (by simp [workHeap, List.append_assoc])
    simpa only [workHeap_append, workHeap_set] using appended
  · exact List.rel_append old (.cons ⟨address, rfl, found⟩ .nil)

private theorem digit_lengths (n d : Nat) (nSingle : n < 10) (dSingle : d < 10)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getN : env "n_rem" = some (text n)) (getD : env "d_rem" = some (text d)) :
    Evaluates program env
      (.boolOp .or
        [.compare (.builtinCall .len [.name "n_rem"]) [(.notEq, .constant (.int 1))],
         .compare (.builtinCall .len [.name "d_rem"]) [(.notEq, .constant (.int 1))]])
      heap (.scalar (.bool false)) heap globals := by
  apply boolean_or _ _ false false
  · apply Evaluates.compare (value := .scalar (.bool false))
      (Evaluates.builtin (.cons (Evaluates.lookup getN) .nil) (by
        change some (Except.ok (ε := ExceptionObject) (number (decimal n).length), heap) =
          some (.ok (number 1), heap)
        rw [(one_digit_facts ⟨n, nSingle⟩).1])) Evaluates.constant
    rfl
  · apply Evaluates.compare (value := .scalar (.bool false))
      (Evaluates.builtin (.cons (Evaluates.lookup getD) .nil) (by
        change some (Except.ok (ε := ExceptionObject) (number (decimal d).length), heap) =
          some (.ok (number 1), heap)
        rw [(one_digit_facts ⟨d, dSingle⟩).1])) Evaluates.constant
    rfl

private theorem zero_text (n : Nat) (single : n < 10)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getN : env "d_rem" = some (text n)) :
    Evaluates program env (.compare (.name "d_rem") [(.eq, .constant (.string "0"))])
      heap (.scalar (.bool (decide (n = 0)))) heap globals := by
  apply Evaluates.compare (Evaluates.lookup getN) Evaluates.constant
  have zero_iff := (one_digit_facts ⟨n, single⟩).2.2
  change (n : Int).repr = "0" ↔ n = 0 at zero_iff
  simp [HeapCalls.compare, scalarCompare, zero_iff]

private abbrev digitValues : List Mutable.Value :=
  "123456789".toList.map (fun char => .scalar (.string (String.singleton char)))

private theorem digitValues_item : ∀ position : Fin 9,
    digitValues[position.val]? = some (text (position.val + 1)) := by decide +kernel

private theorem digitValues_done : digitValues[9]? = none := by decide +kernel

private theorem digitValues_sequence (heap : Mutable.Heap) :
    Mutable.sequence (.scalar (.string "123456789")) heap = some (.ok digitValues, heap) := rfl

private theorem digit_iteration (n d digit : Nat)
    (nLower : 10 ≤ n) (nUpper : n < 100) (dLower : 10 ≤ d) (dUpper : d < 100)
    (positive : 0 < digit) (single : digit < 10)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List (Nat × Nat)) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (locals : SearchLocals env base.length n d)
    (getDigit : env "c" = some (text digit)) :
    ∃ finalValues finalGarbage finalEnv,
      LoopIteration env digitBody (workHeap base values garbage) (cancellation n d digit)
        finalEnv (workHeap base finalValues finalGarbage) globals ∧
      SearchLocals finalEnv base.length n d ∧
      PairValues (workHeap base finalValues finalGarbage) finalValues
        (pairs ++ if cancellation n d digit then [(n, d)] else []) := by
  let heap := workHeap base values garbage
  have common : Evaluates program env
      (.boolOp .and [.membership (.name "c") (.name "ns") false,
        .membership (.name "c") (.name "ds") false])
      heap (.scalar (.bool (decide (hasDigit n digit) && decide (hasDigit d digit)))) heap globals :=
    boolean_and _ _ _ _ env heap globals
      (membership_evaluates n digit nLower nUpper positive single env heap globals "ns"
        locals.numeratorText getDigit)
      (membership_evaluates d digit dLower dUpper positive single env heap globals "ds"
        locals.denominatorText getDigit)
  by_cases present : hasDigit n digit ∧ hasDigit d digit
  · have both : (decide (hasDigit n digit) && decide (hasDigit d digit)) = true := by simp [present.1, present.2]
    rw [both] at common
    let remEnv := setLocal (setLocal env "n_rem" (text (remainder n digit)))
      "d_rem" (text (remainder d digit))
    have remLocals : SearchLocals remEnv base.length n d :=
      (locals.other "n_rem" _ (by decide)).other "d_rem" _ (by decide)
    have remN : remEnv "n_rem" = some (text (remainder n digit)) := by simp [remEnv, setLocal]
    have remD : remEnv "d_rem" = some (text (remainder d digit)) := by simp [remEnv, setLocal]
    have rems : BlockContinues program env (cancellationBody.take 2) heap remEnv heap globals := by
      apply BlockContinues.append
        (BlockContinues.assign (replacement_evaluates n digit nLower nUpper positive single
          present.1 env heap globals "ns" locals.numeratorText getDigit))
      exact BlockContinues.assign (replacement_evaluates d digit dLower dUpper positive single
        present.2 _ heap globals "ds"
        (by simpa [setLocal] using locals.denominatorText) (by simpa [setLocal] using getDigit))
    have lengths := digit_lengths (remainder n digit) (remainder d digit)
      (remainder_lt n digit nUpper) (remainder_lt d digit dUpper) remEnv heap globals remN remD
    have zeroTest := zero_text (remainder d digit) (remainder_lt d digit dUpper)
      remEnv heap globals remD
    by_cases zero : remainder d digit = 0
    · have rejected : cancellation n d digit = false := by simp [cancellation, zero]
      refine ⟨values, garbage, remEnv, ?_, remLocals, ?_⟩
      · rw [rejected]
        apply LoopIteration.branch common rfl
        apply LoopIteration.prepend rems
        apply LoopIteration.branch lengths rfl
        apply LoopIteration.branch (by simpa [zero] using zeroTest) rfl
        apply LoopIteration.skip
        intro next following
        rfl
      · simpa [rejected] using represented
    · let intValues : List Mutable.Value := [number (remainder n digit), number (remainder d digit)]
      let intEnv := assignNames remEnv ["n2", "d2"] intValues
      let intGarbage := garbage ++ [.tuple intValues]
      let intHeap := workHeap base values intGarbage
      have intLocals : SearchLocals intEnv base.length n d := by
        simpa [intEnv, intValues, assignNames] using
          (remLocals.other "n2" (number (remainder n digit)) (by decide)).other
            "d2" (number (remainder d digit)) (by decide)
      have intN : intEnv "n2" = some (number (remainder n digit)) := by
        simp [intEnv, intValues, assignNames, setLocal]
      have intD : intEnv "d2" = some (number (remainder d digit)) := by
        simp [intEnv, intValues, assignNames, setLocal]
      have unpacked : BlockContinues program remEnv (cancellationBody.drop 4 |>.take 1)
          heap intEnv intHeap globals := by
        apply BlockContinues.unpack (names := ["n2", "d2"]) (items := intValues) rfl
          (Evaluates.tuple
            (.cons (digit_integer_evaluates _ (remainder_lt n digit nUpper) remEnv heap globals
              _ (Evaluates.lookup remN))
            (.cons (digit_integer_evaluates _ (remainder_lt d digit dUpper) remEnv heap globals
              _ (Evaluates.lookup remD)) .nil)) (by simp))
        · change Mutable.sequence (.ref heap.length) (heap ++ [.tuple intValues]) =
            some (.ok intValues, intHeap)
          rw [show intHeap = heap ++ [.tuple intValues] by
            simp only [intHeap, intGarbage, heap, workHeap_append]]
          simp [Mutable.sequence]
        · rfl
      have intRepresented : PairValues intHeap values pairs := by
        simpa [intHeap, intGarbage, heap, workHeap_append] using
          represented.append_heap [.tuple intValues]
      have equality : Evaluates program intEnv
          (.compare (.binOp (.name "n") .mult (.name "d2"))
            [(.eq, .binOp (.name "d") .mult (.name "n2"))])
          intHeap (.scalar (.bool (decide (n * remainder d digit = d * remainder n digit))))
          intHeap globals := by
        apply Evaluates.compare (av := number (n * remainder d digit))
          (bv := number (d * remainder n digit)) (middle := intHeap) (afterRight := intHeap)
          (Evaluates.binary (Evaluates.lookup intLocals.numerator) (Evaluates.lookup intD)
            (by simp [Mutable.binary, scalarBinary, number]))
          (Evaluates.binary (Evaluates.lookup intLocals.denominator) (Evaluates.lookup intN)
            (by simp [Mutable.binary, scalarBinary, number]))
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, -Int.natCast_mul]
        exact Int.ofNat_inj
      by_cases equal : n * remainder d digit = d * remainder n digit
      · have accepted : cancellation n d digit = true := by
          simp [cancellation, present.1, present.2, zero, equal]
        obtain ⟨appended, representedFinal⟩ := append_fraction base values intGarbage pairs n d
          intEnv globals intRepresented intLocals.result intLocals.numerator intLocals.denominator
        refine ⟨values ++ [.ref intHeap.length], intGarbage ++ [.tuple [number n, number d]],
          intEnv, ?_, intLocals, ?_⟩
        · rw [accepted]
          apply LoopIteration.branch common rfl
          apply LoopIteration.prepend rems
          apply LoopIteration.branch lengths rfl
          apply LoopIteration.branch (by simpa [zero] using zeroTest) rfl
          apply LoopIteration.prepend unpacked
          apply LoopIteration.branch (by simpa [equal] using equality) rfl
          apply LoopIteration.prepend appended
          apply LoopIteration.stop
          intro next following
          rfl
        · simpa only [accepted, ↓reduceIte] using representedFinal
      · have rejected : cancellation n d digit = false := by simp [cancellation, equal]
        refine ⟨values, intGarbage, intEnv, ?_, intLocals, ?_⟩
        · rw [rejected]
          apply LoopIteration.branch common rfl
          apply LoopIteration.prepend rems
          apply LoopIteration.branch lengths rfl
          apply LoopIteration.branch (by simpa [zero] using zeroTest) rfl
          apply LoopIteration.prepend unpacked
          apply LoopIteration.branch (by simpa [equal] using equality) rfl
          exact LoopIteration.done intEnv intHeap globals
        · simpa [rejected] using intRepresented
  · have rejected : cancellation n d digit = false := by
      simp only [cancellation, decide_eq_false_iff_not]
      exact fun h => present ⟨h.1, h.2.1⟩
    have neither : (decide (hasDigit n digit) && decide (hasDigit d digit)) = false := by
      by_cases hn : hasDigit n digit <;> by_cases hd : hasDigit d digit <;> simp_all
    rw [neither] at common
    refine ⟨values, garbage, env, ?_, locals, ?_⟩
    · rw [rejected]
      exact LoopIteration.branch common rfl (LoopIteration.done env heap globals)
    · simpa [rejected] using represented

private theorem LoopIteration.continues
    (iteration : LoopIteration env body heap false middleEnv middleHeap globals)
    (rest : BlockContinues program middleEnv [next] middleHeap finalEnv finalHeap globals) :
    BlockContinues program env (body ++ [.loopResume next]) heap finalEnv finalHeap globals := by
  intro following anchors stack
  exact (iteration next following anchors stack).trans (rest following anchors stack)

private theorem LoopIteration.breaks
    (iteration : LoopIteration env body heap true finalEnv finalHeap globals) :
    BlockContinues program env (body ++ [.loopResume next]) heap finalEnv finalHeap globals := by
  intro following anchors stack
  exact iteration next following anchors stack

private abbrev digitLoop (position : Nat) : Stmt :=
  .sequenceStepLoop "c" (.scalar (.string "123456789")) position digitBody []

private theorem digit_continuation (next : Stmt) :
    Calls.loopContinuation digitBody next = .loopResume next := rfl

private theorem digits_loop (n d : Nat)
    (nLower : 10 ≤ n) (nUpper : n < 100) (dLower : 10 ≤ d) (dUpper : d < 100)
    (position remaining : Nat) (length : position + remaining = 9)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List (Nat × Nat)) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (locals : SearchLocals env base.length n d) :
    ∃ finalValues finalGarbage finalEnv,
      BlockContinues program env [digitLoop position] (workHeap base values garbage)
        finalEnv (workHeap base finalValues finalGarbage) globals ∧
      SearchLocals finalEnv base.length n d ∧
      PairValues (workHeap base finalValues finalGarbage) finalValues
        (pairs ++ if (List.range' (position + 1) remaining).any (cancellation n d)
          then [(n, d)] else []) := by
  cases remaining with
  | zero =>
      have endPosition : position = 9 := by omega
      subst position
      refine ⟨values, garbage, env, ?_, locals, ?_⟩
      · exact BlockContinues.sequence_done (digitValues_sequence _) digitValues_done .nil
      · simpa using represented
  | succ remaining =>
      have index : position < 9 := by omega
      let iterationEnv := setLocal env "c" (text (position + 1))
      have iterationLocals : SearchLocals iterationEnv base.length n d :=
        locals.other "c" _ (by decide)
      obtain ⟨middleValues, middleGarbage, middleEnv, iteration, middleLocals, middlePairs⟩ :=
        digit_iteration n d (position + 1) nLower nUpper dLower dUpper (by omega) (by omega)
          base values garbage pairs iterationEnv globals represented iterationLocals (by simp [iterationEnv, setLocal])
      cases accepted : cancellation n d (position + 1) with
      | false =>
          have currentPairs : PairValues (workHeap base middleValues middleGarbage) middleValues pairs := by
            simpa [accepted] using middlePairs
          obtain ⟨finalValues, finalGarbage, finalEnv, rest, finalLocals, finalPairs⟩ :=
            digits_loop n d nLower nUpper dLower dUpper (position + 1) remaining (by omega)
              base middleValues middleGarbage pairs middleEnv globals currentPairs middleLocals
          refine ⟨finalValues, finalGarbage, finalEnv, ?_, finalLocals, ?_⟩
          · apply BlockContinues.sequence_step (source := .scalar (.string "123456789"))
              (digitValues_sequence _) (digitValues_item ⟨position, index⟩)
            have resumed := LoopIteration.continues (by simpa only [accepted] using iteration) rest
            simpa only [digit_continuation] using resumed
          · simpa only [List.range'_succ, List.any_cons, accepted, Bool.false_or,
              Nat.add_assoc] using finalPairs
      | true =>
          refine ⟨middleValues, middleGarbage, middleEnv, ?_, middleLocals, ?_⟩
          · apply BlockContinues.sequence_step (source := .scalar (.string "123456789"))
              (digitValues_sequence _) (digitValues_item ⟨position, index⟩)
            have stopped := LoopIteration.breaks (next := digitLoop (position + 1))
              (by simpa only [accepted] using iteration)
            simpa only [digit_continuation] using stopped
          · simpa only [List.range'_succ, List.any_cons, accepted, Bool.true_or] using middlePairs
termination_by remaining

private structure RowLocals (env : HeapCalls.Env) (address n : Nat) : Prop where
  result : env "res" = some (.ref address)
  numerator : env "n" = some (number n)

private theorem RowLocals.other (locals : RowLocals env address n) (name : String)
    (value : Mutable.Value) (unrelated : name ∉ ["res", "n"]) :
    RowLocals (setLocal env name value) address n := by
  rcases locals with ⟨result, numerator⟩
  simp only [List.mem_cons, not_or] at unrelated
  constructor <;> simp_all [setLocal, eq_comm]

private theorem trailing_zero_guard (n d : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getN : env "n" = some (number n))
    (getD : env "d" = some (number d)) :
    Evaluates program env
      (.boolOp .and
        [.compare (.binOp (.name "n") .mod (.constant (.int 10))) [(.eq, .constant (.int 0))],
         .compare (.binOp (.name "d") .mod (.constant (.int 10))) [(.eq, .constant (.int 0))]])
      heap (.scalar (.bool (decide (n % 10 = 0 ∧ d % 10 = 0)))) heap globals := by
  have first : Evaluates program env
      (.compare (.binOp (.name "n") .mod (.constant (.int 10))) [(.eq, .constant (.int 0))])
      heap (.scalar (.bool (decide (n % 10 = 0)))) heap globals := by
    apply Evaluates.compare (av := number (n % 10)) (bv := number 0)
      (middle := heap) (afterRight := heap)
      (Evaluates.binary (Evaluates.lookup getN) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, -Int.natCast_emod]
    exact Int.ofNat_inj
  have second : Evaluates program env
      (.compare (.binOp (.name "d") .mod (.constant (.int 10))) [(.eq, .constant (.int 0))])
      heap (.scalar (.bool (decide (d % 10 = 0)))) heap globals := by
    apply Evaluates.compare (av := number (d % 10)) (bv := number 0)
      (middle := heap) (afterRight := heap)
      (Evaluates.binary (Evaluates.lookup getD) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, -Int.natCast_emod]
    exact Int.ofNat_inj
  have combined := boolean_and _ _ _ _ env heap globals first second
  by_cases hn : n % 10 = 0 <;> by_cases hd : d % 10 = 0 <;>
    simpa [hn, hd] using combined

private theorem denominator_iteration (n d : Nat)
    (nLower : 10 ≤ n) (nUpper : n < 100) (dLower : 10 ≤ d) (dUpper : d < 100)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List (Nat × Nat)) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (locals : RowLocals env base.length n) (getD : env "d" = some (number d)) :
    ∃ finalValues finalGarbage finalEnv,
      LoopIteration env denominatorBody (workHeap base values garbage) false finalEnv
        (workHeap base finalValues finalGarbage) globals ∧
      RowLocals finalEnv base.length n ∧
      PairValues (workHeap base finalValues finalGarbage) finalValues
        (pairs ++ if optimizedPredicate n d then [(n, d)] else []) := by
  let heap := workHeap base values garbage
  have guard := trailing_zero_guard n d env heap globals locals.numerator getD
  by_cases trivial : n % 10 = 0 ∧ d % 10 = 0
  · refine ⟨values, garbage, env, ?_, locals, ?_⟩
    · apply LoopIteration.branch (by simpa [trivial] using guard) rfl
      apply LoopIteration.skip
      intro next following
      rfl
    · simpa [optimizedPredicate, trivial] using represented
  · let strings : List Mutable.Value := [text n, text d]
    let stringEnv := assignNames env ["ns", "ds"] strings
    let stringGarbage := garbage ++ [.tuple strings]
    let stringHeap := workHeap base values stringGarbage
    have stringLocals : SearchLocals stringEnv base.length n d := by
      constructor
      · simpa [stringEnv, strings, assignNames, setLocal] using locals.result
      · simpa [stringEnv, strings, assignNames, setLocal] using locals.numerator
      · simpa [stringEnv, strings, assignNames, setLocal] using getD
      · simp [stringEnv, strings, assignNames, setLocal]
      · simp [stringEnv, strings, assignNames, setLocal]
    have unpacked : BlockContinues program env (denominatorBody.drop 1 |>.take 1)
        heap stringEnv stringHeap globals := by
      apply BlockContinues.unpack (names := ["ns", "ds"]) (items := strings) rfl
        (Evaluates.tuple
          (.cons (decimal_evaluates n env heap globals _ (Evaluates.lookup locals.numerator))
          (.cons (decimal_evaluates d env heap globals _ (Evaluates.lookup getD)) .nil))
          (by simp))
      · change Mutable.sequence (.ref heap.length) (heap ++ [.tuple strings]) =
          some (.ok strings, stringHeap)
        rw [show stringHeap = heap ++ [.tuple strings] by
          simp only [stringHeap, stringGarbage, heap, workHeap_append]]
        simp [Mutable.sequence]
      · rfl
    have stringPairs : PairValues stringHeap values pairs := by
      simpa [stringHeap, stringGarbage, heap, workHeap_append] using
        represented.append_heap [.tuple strings]
    obtain ⟨finalValues, finalGarbage, finalEnv, digits, finalLocals, finalPairs⟩ :=
      digits_loop n d nLower nUpper dLower dUpper 0 9 rfl base values stringGarbage
        pairs stringEnv globals stringPairs stringLocals
    have searched : BlockContinues program stringEnv (denominatorBody.drop 2)
        stringHeap finalEnv (workHeap base finalValues finalGarbage) globals := by
      apply BlockContinues.for_sequence (by intros; simp) Evaluates.constant
        (digitValues_sequence stringHeap)
      exact digits
    refine ⟨finalValues, finalGarbage, finalEnv, ?_,
      ⟨finalLocals.result, finalLocals.numerator⟩, ?_⟩
    · apply LoopIteration.branch (by simpa [trivial] using guard) rfl
      apply LoopIteration.prepend unpacked
      apply LoopIteration.prepend searched
      exact LoopIteration.done finalEnv _ globals
    · rw [show optimizedPredicate n d = (List.range' 1 9).any (cancellation n d) from
        if_neg trivial]
      exact finalPairs

private abbrev denominatorLoop (d : Nat) : Stmt :=
  .rangeStepLoop "d" d 100 1 denominatorBody []

private theorem denominator_continuation (next : Stmt) :
    Calls.loopContinuation denominatorBody next = .loopResume next := rfl

private theorem denominator_loop (n d remaining : Nat)
    (nLower : 10 ≤ n) (nUpper : n < 100) (dLower : 10 ≤ d)
    (length : d + remaining = 100)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List (Nat × Nat)) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (locals : RowLocals env base.length n) :
    ∃ finalValues finalGarbage finalEnv,
      BlockContinues program env [denominatorLoop d] (workHeap base values garbage)
        finalEnv (workHeap base finalValues finalGarbage) globals ∧
      RowLocals finalEnv base.length n ∧
      PairValues (workHeap base finalValues finalGarbage) finalValues
        (pairs ++ ((List.range' d remaining).filter (optimizedPredicate n)).map (n, ·)) := by
  cases remaining with
  | zero =>
      have endD : d = 100 := by omega
      subst d
      refine ⟨values, garbage, env, ?_, locals, ?_⟩
      · exact BlockContinues.range_done rfl .nil
      · simpa using represented
  | succ remaining =>
      have within : d < 100 := by omega
      let iterationEnv := setLocal env "d" (number d)
      obtain ⟨middleValues, middleGarbage, middleEnv, iteration, middleLocals, middlePairs⟩ :=
        denominator_iteration n d nLower nUpper dLower within base values garbage pairs
          iterationEnv globals represented (locals.other "d" _ (by decide))
          (by simp [iterationEnv, setLocal])
      obtain ⟨finalValues, finalGarbage, finalEnv, rest, finalLocals, finalPairs⟩ :=
        denominator_loop n (d + 1) remaining nLower nUpper (by omega) (by omega)
          base middleValues middleGarbage (pairs ++ if optimizedPredicate n d then [(n, d)] else [])
          middleEnv globals middlePairs middleLocals
      refine ⟨finalValues, finalGarbage, finalEnv, ?_, finalLocals, ?_⟩
      · apply BlockContinues.range_step
          (by simp [Calls.rangeContinues]; omega)
        have resumed := LoopIteration.continues iteration rest
        simpa only [denominator_continuation, Int.natCast_add, Int.natCast_one] using resumed
      · cases accepted : optimizedPredicate n d <;>
          simpa [List.range'_succ, List.filter_cons, accepted, List.append_assoc] using finalPairs
termination_by remaining

private theorem numerator_iteration (n : Nat) (lower : 10 ≤ n) (upper : n < 100)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List (Nat × Nat)) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (locals : RowLocals env base.length n) :
    ∃ finalValues finalGarbage finalEnv,
      BlockContinues program env numeratorBody (workHeap base values garbage)
        finalEnv (workHeap base finalValues finalGarbage) globals ∧
      RowLocals finalEnv base.length n ∧
      PairValues (workHeap base finalValues finalGarbage) finalValues (pairs ++ optimizedRow n) := by
  obtain ⟨finalValues, finalGarbage, finalEnv, searched, finalLocals, finalPairs⟩ :=
    denominator_loop n (n + 1) (99 - n) lower upper (by omega) (by omega)
      base values garbage pairs env globals represented locals
  refine ⟨finalValues, finalGarbage, finalEnv, ?_, finalLocals, finalPairs⟩
  apply BlockContinues.for_range (middle := workHeap base values garbage)
    (.cons (Evaluates.binary (middle := workHeap base values garbage)
      (afterRight := workHeap base values garbage) (Evaluates.lookup locals.numerator) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])) (.cons Evaluates.constant .nil))
    (show rangeBounds [number (n + 1), number 100] = .ok ((n + 1 : Nat), 100, 1) from rfl)
  exact searched

private abbrev numeratorLoop (n : Nat) : Stmt :=
  .rangeStepLoop "n" n 100 1 numeratorBody []

private theorem numerator_continuation (next : Stmt) :
    Calls.loopContinuation numeratorBody next = next := rfl

private theorem numerator_loop (n remaining : Nat) (lower : 10 ≤ n) (length : n + remaining = 100)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List (Nat × Nat)) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (represented : PairValues (workHeap base values garbage) values pairs)
    (getResult : env "res" = some (.ref base.length)) :
    ∃ finalValues finalGarbage finalEnv,
      BlockContinues program env [numeratorLoop n] (workHeap base values garbage)
        finalEnv (workHeap base finalValues finalGarbage) globals ∧
      finalEnv "res" = some (.ref base.length) ∧
      PairValues (workHeap base finalValues finalGarbage) finalValues
        (pairs ++ (List.range' n remaining).flatMap optimizedRow) := by
  cases remaining with
  | zero =>
      have endN : n = 100 := by omega
      subst n
      refine ⟨values, garbage, env, ?_, getResult, ?_⟩
      · exact BlockContinues.range_done rfl .nil
      · simpa using represented
  | succ remaining =>
      have upper : n < 100 := by omega
      let iterationEnv := setLocal env "n" (number n)
      have locals : RowLocals iterationEnv base.length n := by
        constructor
        · simpa [iterationEnv, setLocal] using getResult
        · simp [iterationEnv, setLocal]
      obtain ⟨middleValues, middleGarbage, middleEnv, iteration, middleLocals, middlePairs⟩ :=
        numerator_iteration n lower upper base values garbage pairs iterationEnv globals represented locals
      obtain ⟨finalValues, finalGarbage, finalEnv, rest, finalResult, finalPairs⟩ :=
        numerator_loop (n + 1) remaining (by omega) (by omega) base middleValues middleGarbage
          (pairs ++ optimizedRow n) middleEnv globals middlePairs middleLocals.result
      refine ⟨finalValues, finalGarbage, finalEnv, ?_, finalResult, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [numerator_continuation, Int.natCast_add, Int.natCast_one] using iteration.append rest
      · simpa only [List.range'_succ, List.flatMap_cons, List.append_assoc] using finalPairs
termination_by remaining

/-- The source search retains tuple references and all intermediate allocations.
Its returned list is characterized extensionally, without imposing allocation
positions on its elements or assuming a set iteration order. -/
theorem find_curious_fractions.result (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ finalHeap values,
      ReturnsValue program "find_curious_fractions" [] heap (.ref heap.length) finalHeap globals ∧
      finalHeap[heap.length]? = some (.list values) ∧ PairValues finalHeap values fractions ∧
      ∀ (address : Nat) (object : Mutable.Object), heap[address]? = some object → finalHeap[address]? = some object := by
  let initialEnv := functionEnv []
    ((findFunctionIn program.body "find_curious_fractions").getD ([], [])).2 [] globals
  let env := setLocal initialEnv "res" (.ref heap.length)
  obtain ⟨values, garbage, finalEnv, searched, finalResult, represented⟩ :=
    numerator_loop 10 90 (by decide) rfl heap [] [] [] env globals .nil (by simp [env, setLocal])
  refine ⟨workHeap heap values garbage, values, ?_, workHeap_found heap values garbage, ?_, ?_⟩
  · apply BlockReturns.function rfl rfl
    apply BlockContinues.returns (BlockContinues.annAssign (Evaluates.list .nil))
    apply BlockContinues.returns
      (BlockContinues.for_range (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl searched)
    exact BlockReturns.return_value (Evaluates.lookup finalResult)
  · have collectedEquality : (List.range' 10 90).flatMap optimizedRow = fractions := optimized_fractions
    simpa [collectedEquality] using represented
  · intro address object found
    change (heap ++ (.list values :: garbage))[address]? = some object
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found


private theorem PairValues.at {heap : Mutable.Heap} {values : List Mutable.Value}
    {pairs : List (Nat × Nat)} {pair : Nat × Nat} (represented : PairValues heap values pairs)
    (position : Nat) (item : pairs[position]? = some pair) :
    ∃ address, values[position]? = some (.ref address) ∧
      heap[address]? = some (.tuple [number pair.1, number pair.2]) := by
  induction represented generalizing position with
  | nil => simp at item
  | @cons value head values pairs first rest ih =>
      cases position with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at item
          subst pair
          obtain ⟨address, rfl, found⟩ := first
          exact ⟨address, rfl, found⟩
      | succ position =>
          exact ih position (by simpa using item)

private abbrev multiplicationBody : List Stmt :=
  [.augAssign (.name "num_prod") .mult (.name "n"),
   .augAssign (.name "den_prod") .mult (.name "d")]

private def productFrom (items : List (Nat × Nat)) (pn pd : Nat) : Nat × Nat :=
  items.foldl (fun acc item => (acc.1 * item.1, acc.2 * item.2)) (pn, pd)

/- The product loop is generic over the list and its tuple addresses. Only the
small, already certified result list is evaluated in the source helper wrapper. -/
private theorem multiplication_loop (pairs remaining : List (Nat × Nat)) (position : Nat)
    (suffix : pairs.drop position = remaining) (heap : Mutable.Heap) (collector : Nat)
    (values : List Mutable.Value) (found : heap[collector]? = some (.list values))
    (represented : PairValues heap values pairs) (env : HeapCalls.Env) (pn pd : Nat)
    (globals : Mutable.Globals)
    (getN : env "num_prod" = some (number pn)) (getD : env "den_prod" = some (number pd)) :
    ∃ finalEnv, BlockContinues program env
      [.unpackStepLoop ["n", "d"] (.ref collector) position multiplicationBody []]
      heap finalEnv heap globals ∧
      finalEnv "num_prod" = some (number (productFrom remaining pn pd).1) ∧
      finalEnv "den_prod" = some (number (productFrom remaining pn pd).2) := by
  induction remaining generalizing position env pn pd with
  | nil =>
      refine ⟨env, ?_, getN, getD⟩
      apply BlockContinues.unpack_done (source := .ref collector) (items := values) (by simp [sequenceValue, Mutable.sequence, found])
      · have missing : pairs[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := pairs) (i := position)).symm
        have length := List.Forall₂.length_eq represented
        have outside := List.getElem?_eq_none_iff.mp missing
        exact List.getElem?_eq_none (by omega)
      · exact BlockContinues.nil
  | cons pair remaining ih =>
      have item : pairs[position]? = some pair := by
        simpa [suffix] using (List.head?_drop (l := pairs) (i := position)).symm
      obtain ⟨address, selected, fields⟩ := represented.at position item
      let bound := assignNames env ["n", "d"] [number pair.1, number pair.2]
      let afterN := setLocal bound "num_prod" (number (pn * pair.1))
      let afterD := setLocal afterN "den_prod" (number (pd * pair.2))
      have multiplyN : BlockContinues program bound [multiplicationBody[0]'(by decide)]
          heap afterN heap globals := by
        apply BlockContinues.augAssign (left := number pn) (right := number pair.1)
          (value := number (pn * pair.1)) (middle := heap) (finalHeap := heap)
        · simpa [bound, assignNames, setLocal] using getN
        · exact Evaluates.lookup (by simp [bound, assignNames, setLocal])
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      have multiplyD : BlockContinues program afterN [multiplicationBody[1]'(by decide)]
          heap afterD heap globals := by
        apply BlockContinues.augAssign (left := number pd) (right := number pair.2)
          (value := number (pd * pair.2)) (middle := heap) (finalHeap := heap)
        · simpa [afterN, bound, assignNames, setLocal] using getD
        · exact Evaluates.lookup (by simp [afterN, bound, assignNames, setLocal])
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      obtain ⟨finalEnv, rest, finalN, finalD⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) afterD (pn * pair.1) (pd * pair.2)
        (by simp [afterD, afterN, setLocal]) (by simp [afterD, setLocal])
      refine ⟨finalEnv, ?_, finalN, finalD⟩
      apply BlockContinues.unpack_step (source := .ref collector) (items := values)
        (value := .ref address) (values := [number pair.1, number pair.2])
        (by simp [sequenceValue, Mutable.sequence, found]) selected
        (by simp [Mutable.sequence, fields]) rfl
      simpa only [Calls.loopContinuation, show Calls.loopControl multiplicationBody = false from rfl,
        ↓reduceIte] using multiplyN.append (multiplyD.append rest)

theorem reduced_product_denominator.result (heap : Mutable.Heap) (collector : Nat)
    (values : List Mutable.Value) (found : heap[collector]? = some (.list values))
    (represented : PairValues heap values fractions) (globals : Mutable.Globals := []) :
    ReturnsValue program "reduced_product_denominator" [.ref collector] heap (number 100) heap globals := by
  let args := ((findFunctionIn program.body "reduced_product_denominator").getD ([], [])).1
  let body := ((findFunctionIn program.body "reduced_product_denominator").getD ([], [])).2
  let initial := functionEnv args body [.ref collector] globals
  let ready := setLocal (setLocal initial "num_prod" (number 1)) "den_prod" (number 1)
  obtain ⟨finalEnv, multiplied, finalN, finalD⟩ := multiplication_loop
    fractions fractions 0 rfl heap collector values found represented ready 1 1 globals
      (by simp [ready, setLocal]) (by simp [ready, setLocal])
  change finalEnv "num_prod" = some (number 387296) at finalN
  change finalEnv "den_prod" = some (number 38729600) at finalD
  have getFractions : ready "fracs" = some (.ref collector) := by
    cases globals <;> rfl
  have loop : BlockContinues program ready
      [.forIn (.tuple [.name "n", .name "d"]) (.name "fracs") multiplicationBody []]
      heap finalEnv heap globals :=
    BlockContinues.for_unpack (items := values) (by intros; simp) rfl (Evaluates.lookup getFractions)
      (by simp [Mutable.sequence, found]) multiplied
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply loop.returns
  apply BlockReturns.assign (value := number 387296) (middle := heap)
  · exact Evaluates.library_call (.cons (Evaluates.lookup finalN) (.cons (Evaluates.lookup finalD) .nil)) rfl rfl
  · apply BlockReturns.augAssign (left := number 387296) (right := number 387296)
      (value := number 1) (middle := heap) (afterOperation := heap)
      (by simpa [setLocal] using finalN) (Evaluates.lookup (by simp [setLocal])) rfl
    apply BlockReturns.augAssign (left := number 38729600) (right := number 387296)
      (value := number 100) (middle := heap) (afterOperation := heap)
      (by simpa [setLocal] using finalD) (Evaluates.lookup (by simp [setLocal])) rfl
    exact BlockReturns.return_value (Evaluates.lookup (by simp [setLocal]))


private abbrev fractionTuple (n d : Nat) : Mutable.Object := .tuple [number n, number d]
private abbrev fractionKey (n d : Nat) : HashKey := .tuple [.scalar (.int n), .scalar (.int d)]
private abbrev fourEntries (a b c d : Nat) : List (HashKey × Mutable.Value) :=
  [(fractionKey 16 64, .ref a), (fractionKey 19 95, .ref b),
   (fractionKey 26 65, .ref c), (fractionKey 49 98, .ref d)]

private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem PairValues.four {heap : Mutable.Heap} {values : List Mutable.Value}
    (represented : PairValues heap values fractions) :
    ∃ a b c d, values = [.ref a, .ref b, .ref c, .ref d] ∧
      heap[a]? = some (fractionTuple 16 64) ∧ heap[b]? = some (fractionTuple 19 95) ∧
      heap[c]? = some (fractionTuple 26 65) ∧ heap[d]? = some (fractionTuple 49 98) := by
  obtain ⟨a, aItem, aFound⟩ := represented.at 0 rfl
  obtain ⟨b, bItem, bFound⟩ := represented.at 1 rfl
  obtain ⟨c, cItem, cFound⟩ := represented.at 2 rfl
  obtain ⟨d, dItem, dFound⟩ := represented.at 3 rfl
  have length : values.length = 4 := List.Forall₂.length_eq represented
  refine ⟨a, b, c, d, ?_, aFound, bFound, cFound, dFound⟩
  apply List.ext_getElem?
  intro index
  match index with
  | 0 => exact aItem
  | 1 => exact bItem
  | 2 => exact cItem
  | 3 => exact dItem
  | index + 4 => simp [List.getElem?_eq_none (by omega : values.length ≤ index + 4)]

/- Set construction retains each original tuple representative. The only new
object is the fresh set; hashes read the tuples without altering their aliases. -/
private theorem allocate_four (heap : Mutable.Heap) (a b c d : Nat)
    (aFound : heap[a]? = some (fractionTuple 16 64))
    (bFound : heap[b]? = some (fractionTuple 19 95))
    (cFound : heap[c]? = some (fractionTuple 26 65))
    (dFound : heap[d]? = some (fractionTuple 49 98)) :
    Mutable.allocateSetValues [.ref a, .ref b, .ref c, .ref d] heap =
      some (.ok (.ref heap.length), heap ++ [.keySet (fourEntries a b c d)]) := by
  have aKey (object : Mutable.Object) := Mutable.hashKey_tuple_scalars
    (heap ++ [object]) a [.int 16, .int 64] (found_append aFound [object])
  have bKey (object : Mutable.Object) := Mutable.hashKey_tuple_scalars
    (heap ++ [object]) b [.int 19, .int 95] (found_append bFound [object])
  have cKey (object : Mutable.Object) := Mutable.hashKey_tuple_scalars
    (heap ++ [object]) c [.int 26, .int 65] (found_append cFound [object])
  have dKey (object : Mutable.Object) := Mutable.hashKey_tuple_scalars
    (heap ++ [object]) d [.int 49, .int 98] (found_append dFound [object])
  unfold Mutable.allocateSetValues
  split
  · next impossible => cases impossible
  · simp [Mutable.allocateSet, List.mapM_cons, Mutable.addItem,
      aKey, bKey, cKey, dKey, List.set_append_right, Mutable.insertKey, Mutable.keyContains,
      fourEntries, fractionKey, HashKey.equal, HashKey.equalFuel, HashKey.depth, HashKey.listDepth]
    rfl

private abbrev literalTuples : Mutable.Heap :=
  [fractionTuple 16 64, fractionTuple 19 95, fractionTuple 26 65, fractionTuple 49 98]

private abbrev literalSet : Expr := .builtinCall .setDisplay
  [.tuple [.constant (.int 16), .constant (.int 64)],
   .tuple [.constant (.int 19), .constant (.int 95)],
   .tuple [.constant (.int 26), .constant (.int 65)],
   .tuple [.constant (.int 49), .constant (.int 98)]]

private theorem literal_set (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    Evaluates program env literalSet heap (.ref (heap.length + 4))
      (heap ++ literalTuples ++ [.keySet
        (fourEntries heap.length (heap.length + 1) (heap.length + 2) (heap.length + 3))]) globals := by
  apply Evaluates.builtin
    (.cons (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp))
    (.cons (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp))
    (.cons (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp))
    (.cons (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp)) .nil))))
  simpa [primitive, literalTuples, fractionTuple, number, List.append_assoc] using
    allocate_four (heap ++ literalTuples) heap.length (heap.length + 1) (heap.length + 2) (heap.length + 3)
      (by simp [literalTuples]) (by simp [literalTuples])
      (by simp [literalTuples]) (by simp [literalTuples])

private theorem asserted_fractions (env : HeapCalls.Env) (heap : Mutable.Heap)
    (collector : Nat) (values : List Mutable.Value) (globals : Mutable.Globals)
    (lookup : env "fracs" = some (.ref collector))
    (found : heap[collector]? = some (.list values)) (represented : PairValues heap values fractions) :
    ∃ extra, Evaluates program env
      (.compare (.builtinCall .set [.name "fracs"]) [(.eq, literalSet)])
      heap (.scalar (.bool true)) (heap ++ extra) globals := by
  obtain ⟨a, b, c, d, rfl, aFound, bFound, cFound, dFound⟩ := represented.four
  let leftSet := Mutable.Object.keySet (fourEntries a b c d)
  let middle := heap ++ [leftSet]
  let rightSet := Mutable.Object.keySet
    (fourEntries middle.length (middle.length + 1) (middle.length + 2) (middle.length + 3))
  let finalHeap := middle ++ literalTuples ++ [rightSet]
  have left : Evaluates program env (.builtinCall .set [.name "fracs"])
      heap (.ref heap.length) middle globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
    simpa [primitive, Mutable.setFrom, Mutable.sequence, found, middle, leftSet] using
      allocate_four heap a b c d aFound bFound cFound dFound
  have right : Evaluates program env literalSet middle (.ref (middle.length + 4)) finalHeap globals :=
    literal_set env middle globals
  have leftFound : finalHeap[heap.length]? = some leftSet := by
    simp [finalHeap, middle, literalTuples, List.append_assoc]
  have rightFound : finalHeap[middle.length + 4]? = some rightSet := by
    simp [finalHeap, literalTuples]
  refine ⟨[leftSet] ++ literalTuples ++ [rightSet], ?_⟩
  have checked : Evaluates program env
      (.compare (.builtinCall .set [.name "fracs"]) [(.eq, literalSet)])
      heap (.scalar (.bool true)) finalHeap globals := by
    apply Evaluates.compare left right
    simp [HeapCalls.compare, leftFound, rightFound, leftSet, rightSet, Mutable.keyEntries]
    rfl
  simpa only [finalHeap, middle, List.append_assoc] using checked

theorem solve.result (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ finalHeap, ReturnsValue program "solve" [] heap (number 100) finalHeap globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let initial := functionEnv args body [] globals
  let ready := setLocal initial "fracs" (.ref heap.length)
  obtain ⟨collectedHeap, values, collected, found, represented, preserved⟩ :=
    find_curious_fractions.result heap globals
  have getFractions : ready "fracs" = some (.ref heap.length) := by simp [ready, setLocal]
  have length : values.length = 4 := List.Forall₂.length_eq represented
  obtain ⟨extra, assertion⟩ := asserted_fractions ready collectedHeap heap.length values globals
    getFractions found represented
  have product := reduced_product_denominator.result (collectedHeap ++ extra) heap.length values
    (found_append found extra) (represented.append_heap extra) globals
  refine ⟨collectedHeap ++ extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.call .nil collected)
  apply BlockReturns.assertion
    (Evaluates.compare (av := number 4) (bv := number 4)
      (middle := collectedHeap) (afterRight := collectedHeap)
      (Evaluates.builtin (.cons (Evaluates.lookup getFractions) .nil)
        (by simp [primitive, Mutable.length, found, length])) Evaluates.constant rfl) rfl
  apply BlockReturns.assertion assertion rfl
  apply BlockReturns.assign (Evaluates.call (.cons (Evaluates.lookup getFractions) .nil) product)
  exact BlockReturns.return_value (Evaluates.lookup (by simp [setLocal]))

/- The two heap interpreters terminate before any set iteration. Lifting their
traces establishes the family property for every possible choice schedule. -/
theorem solve.«Equiv(naive)» :
    (SetCalls.call program "solve" [] []).Equiv (SetCalls.call P33Ast.module "naive" [] []) := by
  intro leftChoices rightChoices
  obtain ⟨leftHeap, optimized⟩ := solve.result []
  obtain ⟨rightHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p33.naive.result []
  rw [reference_value] at reference
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [] []).run leftChoices)
    (b := (SetCalls.call P33Ast.module "naive" [] []).run rightChoices)
    (SetCalls.finishes_of_heap optimized.finishes leftChoices 0)
    (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap (number 100) []
    (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem solve.«Supported» :
    (SetCalls.call program "solve" [] []).Supported ∧ (SetCalls.call P33Ast.module "naive" [] []).Supported := by
  constructor
  · intro choices
    obtain ⟨_, returned⟩ := solve.result []
    exact Execution.supported_of_finishes (SetCalls.finishes_of_heap returned.finishes choices 0)
  · intro choices
    obtain ⟨_, returned⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p33.naive.result []
    exact Execution.supported_of_finishes (SetCalls.finishes_of_heap returned.finishes choices 0)

end solvers.«33»
