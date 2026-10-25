import lang.py.FormalSW.SumCalls
import lang.py.FormalSW.IntegerParsing
import ProjectEulerStatements.Lib.DecimalDigits

namespace ProjectEulerSolutions.DecimalDigitSumCalls
open FormalSW.PyAst HeapCalls
set_option Elab.async false

private def digit (c : Char) : Nat := c.toNat - '0'.toNat

private theorem character_sum (values : List Nat) (valid : ∀ value ∈ values, value < 10) (initial : Nat) :
    (values.map Nat.digitChar).foldl (fun total c => total + digit c) initial = initial + values.sum := by
  induction values generalizing initial with
  | nil => simp
  | cons head rest ih =>
      have first : digit (Nat.digitChar head) = head :=
        ProjectEulerStatements.DecimalDigits.digitChar_val head (valid head (by simp))
      simp only [List.map_cons, List.foldl_cons, first]
      rw [ih (by intro value member; exact valid value (by simp [member]))]
      simp [Nat.add_assoc]

theorem decimal_sum (n : Nat) :
    (toString (n : Int)).toList.foldl (fun total c => total + digit c) 0 = (Nat.digits 10 n).sum := by
  by_cases zero : n = 0
  · subst n; rfl
  · have encoded : (toString (n : Int)).toList = (Nat.digits 10 n).reverse.map Nat.digitChar := by
      simpa [toString, Int.repr, Nat.repr] using
        ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n (by omega)
    rw [encoded, character_sum]
    · simp
    · intro value member
      exact Nat.digits_lt_base (by decide) (by simpa using member)

private theorem parse_digit (c : Char) (valid : c.isDigit) :
    Calls.integerConversion [.string (String.singleton c)] = some (.ok (.int (digit c))) := by
  change Calls.integerConversion [.string (String.ofList [c])] = _
  rw [Calls.integerConversion, FormalSW.Py.Builtin.Int.parseText_digits [c] (by simp)
    (by intro d member; simp only [List.mem_singleton] at member; subst d; exact valid)]
  simp [Calls.parsedInteger, Nat.ofDigitChars, digit]

private theorem cast_fold (characters : List Char) (initial : Nat) :
    characters.foldl (fun (total : Int) c => total + (digit c : Int)) initial =
      ((characters.foldl (fun total c => total + digit c) initial : Nat) : Int) := by
  induction characters generalizing initial with
  | nil => rfl
  | cons head rest ih => simpa only [List.foldl_cons, Int.natCast_add] using ih (initial + digit head)

set_option maxHeartbeats 1000000 in
-- Attached character subtypes and generator continuations require additional elaboration steps.
/-- Parsing and summing an integer's decimal characters computes its positional digit sum. -/
theorem string_sum (program : Module) (env : HeapCalls.Env) (name : String) (n : Nat)
    (heap : Mutable.Heap) (globals : Mutable.Globals := [])
    (get : env name = some (.scalar (.int n))) :
    Evaluates program env
      (.sumGenerator (.builtinCall .int [.name "c"])
        [("c", .builtinCall .str [.name name], [])] (.constant (.int 0)))
      heap (.scalar (.int (Nat.digits 10 n).sum)) heap globals := by
  let characters := (toString (n : Int)).toList
  have total : characters.attach.foldl (fun (total : Int) c => total + (digit c.val : Int)) 0 =
      ((Nat.digits 10 n).sum : Int) := by
    rw [List.foldl_attach (f := fun (total : Int) (c : Char) => total + (digit c : Int))]
    exact (cast_fold characters 0).trans (congrArg (fun value : Nat => (value : Int)) (decimal_sum n))
  rw [← total]
  apply Evaluates.sum_generator_sequence
    (items := characters.map (fun c => .scalar (.string (String.singleton c)))) (by intros; simp)
    (Evaluates.builtin (name := .str) (values := [.scalar (.int n)])
      (value := .scalar (.string (toString (n : Int)))) (.cons (Evaluates.lookup get) .nil) rfl)
    (by simp [Mutable.sequence, characters]) Evaluates.constant rfl
  apply sum_map_loop (values := characters.attach)
    (encode := fun c => .scalar (.string (String.singleton c.val)))
    (number := fun c => (digit c.val : Int)) characters.attach (by decide)
  · simp [sequenceValue, sequenceSource, Mutable.sequence, characters]
  · intro localEnv c lookup
    apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
    change scalarResult (Calls.integerConversion [.string (String.singleton c.val)]) heap = _
    have member : c.val ∈ Nat.toDigits 10 n := by
      simpa [characters, toString, Int.repr, Nat.repr] using c.property
    rw [parse_digit c.val (Nat.isDigit_of_mem_toDigits (by decide) (by decide) member)]
    rfl
  · rfl
  · simp [setLocal]

end ProjectEulerSolutions.DecimalDigitSumCalls
