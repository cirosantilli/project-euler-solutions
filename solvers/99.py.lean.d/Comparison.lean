import solvers.«99.py.lean.d».Refinement
import ProjectEulerSolutions.Lib.SignedPowerComparison

namespace solvers.«99».compare_powers
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals
set_option Elab.async false

abbrev reduction : List Stmt :=
  [.assign [.name "common"] (.call (.name "gcd") [.name "left_exp", .name "right_exp"]),
   .augAssign (.name "left_exp") .floorDiv (.name "common"),
   .augAssign (.name "right_exp") .floorDiv (.name "common"),
   .assign [.name "precision"] (.constant (.int 16)), loop]
abbrev signNormalization : List Stmt :=
  [.assign [.name "direction"] (.constant (.int 1)),
   .ifThen (.compare (.name "left_exp") [(.lt, .constant (.int 0))])
     [.assign [.name "left_exp"] (.unaryOp .uSub (.name "left_exp")),
      .assign [.name "right_exp"] (.unaryOp .uSub (.name "right_exp")),
      .assign [.name "direction"] (.unaryOp .uSub (.constant (.int 1)))] []]
abbrev guards : List Stmt :=
  [.ifThen (.boolOp .and [.compare (.name "left_exp") [(.eq, .constant (.int 0))],
     .compare (.name "right_exp") [(.eq, .constant (.int 0))]]) [.return (.constant (.int 0))] [],
   .ifThen (.boolOp .and [.compare (.name "left_exp") [(.ltE, .constant (.int 0))],
     .compare (.name "right_exp") [(.gtE, .constant (.int 0))]]) [.return (.unaryOp .uSub (.constant (.int 1)))] [],
   .ifThen (.boolOp .and [.compare (.name "left_exp") [(.gtE, .constant (.int 0))],
     .compare (.name "right_exp") [(.ltE, .constant (.int 0))]]) [.return (.constant (.int 1))] [],
   .ifThen (.compare (.name "left_base") [(.eq, .name "right_base")])
     [.return (.binOp (.compare (.name "left_exp") [(.gt, .name "right_exp")]) .sub
       (.compare (.name "left_exp") [(.lt, .name "right_exp")]))] []]
abbrev baseNormalization : List Stmt :=
  [.ifThen (.compare (.name "left_base") [(.eq, .constant (.int 1))])
     [.assign [.name "left_exp"] (.constant (.int 0))] [],
   .ifThen (.compare (.name "right_base") [(.eq, .constant (.int 1))])
     [.assign [.name "right_exp"] (.constant (.int 0))] []]

theorem source_body : executableBody ((findFunctionIn program.body "compare_powers").getD ([], [])).2 =
    baseNormalization ++ guards ++ signNormalization ++ reduction := rfl

structure Inputs (env : HeapCalls.Env) (a : Nat) (m : Int) (b : Nat) (n : Int) : Prop where
  getA : env "left_base" = some (number a)
  getM : env "left_exp" = some (number m)
  getB : env "right_base" = some (number b)
  getN : env "right_exp" = some (number n)

theorem Inputs.set_direction {env : HeapCalls.Env} {a b : Nat} {m n : Int}
    (inputs : Inputs env a m b n) (direction : Int) :
    Inputs (setLocal env "direction" (number direction)) a m b n := by
  rcases inputs with ⟨ha, hm, hb, hn⟩
  constructor <;> simp_all [setLocal]

theorem and_result {env : HeapCalls.Env} {heap : Mutable.Heap} {globals : Mutable.Globals}
    {left right : Expr} {p q : Bool}
    (first : Evaluates program env left heap (.scalar (.bool p)) heap globals)
    (second : Evaluates program env right heap (.scalar (.bool q)) heap globals) :
    Evaluates program env (.boolOp .and [left, right]) heap (.scalar (.bool (p && q))) heap globals := by
  cases p with
  | false => exact Evaluates.boolean_stop (flag := false) first rfl rfl
  | true =>
      exact Evaluates.boolean_next (op := .and) (flag := true) first rfl rfl
        (Evaluates.boolean_last second rfl)

theorem reduction_result (a m b n : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (positiveM : 0 < m) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (direction : Int) (inputs : Inputs env a m b n) (getDirection : env "direction" = some (number direction)) :
    ∃ extra, BlockReturns program env reduction heap
      (number (direction * comparison (a ^ (m / m.gcd n)) (b ^ (n / m.gcd n)))) (heap ++ extra) globals := by
  let commonEnv := setLocal env "common" (number (m.gcd n))
  let leftEnv := setLocal commonEnv "left_exp" (number (m / m.gcd n : Nat))
  let rightEnv := setLocal leftEnv "right_exp" (number (n / m.gcd n : Nat))
  let readyEnv := setLocal rightEnv "precision" (number 16)
  have locals : Locals readyEnv a (m / m.gcd n) b (n / m.gcd n) 16 direction := by
    rcases inputs with ⟨ha, hm, hb, hn⟩
    constructor <;> simp_all [readyEnv, rightEnv, leftEnv, commonEnv, setLocal]
  obtain ⟨extra, ran⟩ := loop_correct a (m / m.gcd n) b (n / m.gcd n) positiveA positiveB
    16 (by decide) direction readyEnv heap globals locals
  have positiveCommon : 0 < m.gcd n := Nat.gcd_pos_of_pos_left n positiveM
  refine ⟨extra, ?_⟩
  apply BlockReturns.assign (value := number (m.gcd n)) (middle := heap)
  · apply Evaluates.library_call
      (.cons (Evaluates.lookup inputs.getM) (.cons (Evaluates.lookup inputs.getN) .nil)) rfl
    change scalarResult (some (Math.gcdCall [.int m, .int n])) heap = _
    rw [Math.gcdCall_pair]
    simp [scalarResult, number]
  apply BlockReturns.augAssign (middle := heap) (afterOperation := heap)
    (left := number m) (right := number (m.gcd n)) (value := number (m / m.gcd n : Nat))
    (by simpa [setLocal] using inputs.getM) (Evaluates.lookup rfl)
  · simp [inplace, Mutable.binary, scalarBinary, number, Nat.ne_of_gt positiveCommon, positiveM]
  apply BlockReturns.augAssign (middle := heap) (afterOperation := heap)
    (left := number n) (right := number (m.gcd n)) (value := number (n / m.gcd n : Nat))
    (by simpa [setLocal] using inputs.getN) (Evaluates.lookup rfl)
  · simp [inplace, Mutable.binary, scalarBinary, number, Nat.ne_of_gt positiveCommon, positiveM]
  apply BlockReturns.assign Evaluates.constant
  exact ran

theorem sign_result (a : Nat) (m : Int) (b : Nat) (n : Int)
    (positiveA : 0 < a) (positiveB : 0 < b)
    (sameSign : (0 < m ∧ 0 < n) ∨ (m < 0 ∧ n < 0))
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) (inputs : Inputs env a m b n) :
    ∃ extra, BlockReturns program env (signNormalization ++ reduction) heap
      (number ((if m < 0 then -1 else 1) * comparison
        (a ^ ((if m < 0 then -m else m).toNat /
          (if m < 0 then -m else m).toNat.gcd (if m < 0 then -n else n).toNat))
        (b ^ ((if m < 0 then -n else n).toNat /
          (if m < 0 then -m else m).toNat.gcd (if m < 0 then -n else n).toNat)))) (heap ++ extra) globals := by
  let initial := setLocal env "direction" (number 1)
  have initialInputs := inputs.set_direction 1
  have test : Evaluates program initial (.compare (.name "left_exp") [(.lt, .constant (.int 0))])
      heap (.scalar (.bool (decide (m < 0)))) heap globals :=
    Evaluates.compare (Evaluates.lookup initialInputs.getM) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases negative : m < 0
  · let leftEnv := setLocal initial "left_exp" (number (-m))
    let rightEnv := setLocal leftEnv "right_exp" (number (-n))
    let readyEnv := setLocal rightEnv "direction" (number (-1))
    have readyInputs : Inputs readyEnv a (-m).toNat b (-n).toNat := by
      rcases inputs with ⟨ha, hm, hb, hn⟩
      constructor <;> simp_all [readyEnv, rightEnv, leftEnv, initial, setLocal]
      all_goals omega
    obtain ⟨extra, ran⟩ := reduction_result a (-m).toNat b (-n).toNat positiveA positiveB
      (by omega) readyEnv heap globals (-1) readyInputs rfl
    refine ⟨extra, ?_⟩
    simp only [if_pos negative]
    apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.branch (flag := true) (by simpa [negative] using test) rfl
    apply BlockReturns.assign (value := number (-m)) (middle := heap)
      (Evaluates.unary (Evaluates.lookup initialInputs.getM) (by rfl))
    apply BlockReturns.assign (value := number (-n)) (middle := heap)
      (Evaluates.unary (Evaluates.lookup (by simpa [setLocal] using initialInputs.getN)) (by rfl))
    apply BlockReturns.assign (value := number (-1)) (middle := heap)
      (Evaluates.unary Evaluates.constant (by rfl))
    exact ran
  · have readyInputs : Inputs initial a m.toNat b n.toNat := by
      simpa [Int.toNat_of_nonneg (by omega : 0 ≤ m), Int.toNat_of_nonneg (by omega : 0 ≤ n)] using initialInputs
    obtain ⟨extra, ran⟩ := reduction_result a m.toNat b n.toNat positiveA positiveB
      (by omega) initial heap globals 1 readyInputs rfl
    refine ⟨extra, ?_⟩
    simp only [if_neg negative]
    apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.branch (flag := false) (by simpa [negative] using test) rfl
    exact ran

theorem guards_result (a : Nat) (m : Int) (b : Nat) (n : Int)
    (positiveA : 0 < a) (positiveB : 0 < b)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) (inputs : Inputs env a m b n) :
    ∃ extra, BlockReturns program env (guards ++ signNormalization ++ reduction) heap
      (number (normalizedComparison a m b n)) (heap ++ extra) globals := by
  have first : Evaluates program env (.boolOp .and
      [.compare (.name "left_exp") [(.eq, .constant (.int 0))],
       .compare (.name "right_exp") [(.eq, .constant (.int 0))]])
      heap (.scalar (.bool (decide (m = 0 ∧ n = 0)))) heap globals := by
    rw [Bool.decide_and]
    apply and_result
    · apply Evaluates.compare (Evaluates.lookup inputs.getM) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · apply Evaluates.compare (Evaluates.lookup inputs.getN) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases zero : m = 0 ∧ n = 0
  · refine ⟨[], ?_⟩
    simp only [normalizedComparison, if_pos zero, List.append_nil]
    apply BlockReturns.branch (flag := true) (by simpa [zero] using first) rfl
    exact BlockReturns.return_value Evaluates.constant
  have second : Evaluates program env (.boolOp .and
      [.compare (.name "left_exp") [(.ltE, .constant (.int 0))],
       .compare (.name "right_exp") [(.gtE, .constant (.int 0))]])
      heap (.scalar (.bool (decide (m ≤ 0 ∧ 0 ≤ n)))) heap globals := by
    rw [Bool.decide_and]
    apply and_result
    · apply Evaluates.compare (Evaluates.lookup inputs.getM) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · apply Evaluates.compare (Evaluates.lookup inputs.getN) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases opposite : m ≤ 0 ∧ 0 ≤ n
  · refine ⟨[], ?_⟩
    simp only [normalizedComparison, if_neg zero, if_pos opposite, List.append_nil]
    apply BlockReturns.branch (flag := false) (by simpa [zero] using first) rfl
    apply BlockReturns.branch (flag := true) (by simpa [opposite] using second) rfl
    exact BlockReturns.return_value (Evaluates.unary Evaluates.constant (by rfl))
  have third : Evaluates program env (.boolOp .and
      [.compare (.name "left_exp") [(.gtE, .constant (.int 0))],
       .compare (.name "right_exp") [(.ltE, .constant (.int 0))]])
      heap (.scalar (.bool (decide (0 ≤ m ∧ n ≤ 0)))) heap globals := by
    rw [Bool.decide_and]
    apply and_result
    · apply Evaluates.compare (Evaluates.lookup inputs.getM) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · apply Evaluates.compare (Evaluates.lookup inputs.getN) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases opposite' : 0 ≤ m ∧ n ≤ 0
  · refine ⟨[], ?_⟩
    simp only [normalizedComparison, if_neg zero, if_neg opposite, if_pos opposite', List.append_nil]
    apply BlockReturns.branch (flag := false) (by simpa [zero] using first) rfl
    apply BlockReturns.branch (flag := false) (by simpa [opposite] using second) rfl
    apply BlockReturns.branch (flag := true) (by simpa [opposite'] using third) rfl
    exact BlockReturns.return_value Evaluates.constant
  have fourth : Evaluates program env (.compare (.name "left_base") [(.eq, .name "right_base")])
      heap (.scalar (.bool (decide (a = b)))) heap globals :=
    Evaluates.compare (Evaluates.lookup inputs.getA) (Evaluates.lookup inputs.getB)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases same : a = b
  · refine ⟨[], ?_⟩
    simp only [normalizedComparison, if_neg zero, if_neg opposite, if_neg opposite', if_pos same, List.append_nil]
    apply BlockReturns.branch (flag := false) (by simpa [zero] using first) rfl
    apply BlockReturns.branch (flag := false) (by simpa [opposite] using second) rfl
    apply BlockReturns.branch (flag := false) (by simpa [opposite'] using third) rfl
    apply BlockReturns.branch (flag := true) (by simpa [same] using fourth) rfl
    exact BlockReturns.return_value (comparison_expression env heap globals "left_exp" "right_exp" m n
      inputs.getM inputs.getN)
  obtain ⟨extra, ran⟩ := sign_result a m b n positiveA positiveB (by omega) env heap globals inputs
  refine ⟨extra, ?_⟩
  simp only [normalizedComparison, if_neg zero, if_neg opposite, if_neg opposite', if_neg same]
  apply BlockReturns.branch (flag := false) (by simpa [zero] using first) rfl
  apply BlockReturns.branch (flag := false) (by simpa [opposite] using second) rfl
  apply BlockReturns.branch (flag := false) (by simpa [opposite'] using third) rfl
  apply BlockReturns.branch (flag := false) (by simpa [same] using fourth) rfl
  exact ran

theorem normalize_result (a : Nat) (m : Int) (b : Nat) (n : Int)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) (inputs : Inputs env a m b n) :
    ∃ readyEnv, BlockContinues program env baseNormalization heap readyEnv heap globals ∧
      Inputs readyEnv a (if a = 1 then 0 else m) b (if b = 1 then 0 else n) := by
  let leftEnv := if a = 1 then setLocal env "left_exp" (number 0) else env
  let readyEnv := if b = 1 then setLocal leftEnv "right_exp" (number 0) else leftEnv
  have leftInputs : Inputs leftEnv a (if a = 1 then 0 else m) b n := by
    rcases inputs with ⟨ha, hm, hb, hn⟩
    by_cases one : a = 1 <;> constructor <;> simp_all [leftEnv, setLocal]
  have readyInputs : Inputs readyEnv a (if a = 1 then 0 else m) b (if b = 1 then 0 else n) := by
    rcases leftInputs with ⟨ha, hm, hb, hn⟩
    by_cases one : b = 1 <;> constructor <;> simp_all [readyEnv, setLocal]
  refine ⟨readyEnv, ?_, readyInputs⟩
  apply BlockContinues.append (body := baseNormalization.take 1) (rest := baseNormalization.drop 1)
    (middleEnv := leftEnv) (middleHeap := heap)
  · have test : Evaluates program env (.compare (.name "left_base") [(.eq, .constant (.int 1))])
        heap (.scalar (.bool (decide (a = 1)))) heap globals :=
      Evaluates.compare (Evaluates.lookup inputs.getA) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    by_cases one : a = 1
    · apply BlockContinues.branch (flag := true) (by simpa [one] using test) rfl
      simpa [leftEnv, one] using (BlockContinues.assign (name := "left_exp")
        (module := program) (env := env) (globals := globals) (heap := heap)
        (Evaluates.constant (value := .int 0)))
    · apply BlockContinues.branch (flag := false) (by simpa [one] using test) rfl
      simpa [leftEnv, one] using (BlockContinues.nil (module := program) (env := env) (globals := globals) (heap := heap))
  · have test : Evaluates program leftEnv (.compare (.name "right_base") [(.eq, .constant (.int 1))])
        heap (.scalar (.bool (decide (b = 1)))) heap globals :=
      Evaluates.compare (Evaluates.lookup leftInputs.getB) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    by_cases one : b = 1
    · apply BlockContinues.branch (flag := true) (by simpa [one] using test) rfl
      simpa [readyEnv, one] using (BlockContinues.assign (name := "right_exp")
        (module := program) (env := leftEnv) (globals := globals) (heap := heap)
        (Evaluates.constant (value := .int 0)))
    · apply BlockContinues.branch (flag := false) (by simpa [one] using test) rfl
      simpa [readyEnv, one] using (BlockContinues.nil (module := program) (env := leftEnv) (globals := globals) (heap := heap))

theorem result (a : Nat) (m : Int) (b : Nat) (n : Int) (positiveA : 0 < a) (positiveB : 0 < b)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "compare_powers" [number a, number m, number b, number n] heap
      (number (signedComparison a m b n)) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "compare_powers").getD ([], [])).1
  let source := ((findFunctionIn program.body "compare_powers").getD ([], [])).2
  let entry := functionEnv args source [number a, number m, number b, number n] globals
  have inputs : Inputs entry a m b n := by constructor <;> cases globals <;> rfl
  obtain ⟨readyEnv, initialized, readyInputs⟩ := normalize_result a m b n entry heap globals inputs
  obtain ⟨extra, ran⟩ := guards_result a (if a = 1 then 0 else m) b (if b = 1 then 0 else n)
    positiveA positiveB readyEnv heap globals readyInputs
  change BlockReturns program readyEnv (guards ++ signNormalization ++ reduction) heap
    (number (comparisonAlgorithm a m b n)) (heap ++ extra) globals at ran
  rw [comparisonAlgorithm_eq a m b n positiveA positiveB] at ran
  refine ⟨extra, ?_⟩
  apply BlockReturns.function (args := args) (body := source) rfl rfl
  change BlockReturns program entry (baseNormalization ++ guards ++ signNormalization ++ reduction) heap
    (number (signedComparison a m b n)) (heap ++ extra) globals
  simpa only [List.append_assoc] using initialized.returns ran

end solvers.«99».compare_powers
