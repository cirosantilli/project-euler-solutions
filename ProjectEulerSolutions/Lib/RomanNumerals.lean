import Mathlib.Data.List.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic

/-! Adjacent-subtractive parsing and greedy Roman-token encoding, with no bound on input size. -/
namespace ProjectEulerSolutions.RomanNumerals
set_option Elab.async false

def value (symbol : String) : Nat :=
  if symbol = "I" then 1 else if symbol = "V" then 5 else if symbol = "X" then 10
  else if symbol = "L" then 50 else if symbol = "C" then 100 else if symbol = "D" then 500
  else if symbol = "M" then 1000 else 0

def symbols : List String := ["I", "V", "X", "L", "C", "D", "M"]

theorem value_positive (symbol : String) (valid : symbol ∈ symbols) : 0 < value symbol := by
  simp only [symbols, List.mem_cons, List.not_mem_nil, or_false] at valid
  rcases valid with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [value]

def parse : List String → Nat
  | [] => 0
  | [first] => value first
  | first :: second :: rest =>
      if value first < value second then value second - value first + parse rest
      else value first + parse (second :: rest)

def parseAcc : List String → Nat → Nat
  | [], total => total
  | [first], total => total + value first
  | first :: second :: rest, total =>
      if value first < value second then parseAcc rest (total + (value second - value first))
      else parseAcc (second :: rest) (total + value first)

theorem parseAcc_eq (input : List String) (total : Nat) :
    parseAcc input total = total + parse input := by
  cases input with
  | nil => simp [parseAcc, parse]
  | cons first rest =>
      cases rest with
      | nil => rfl
      | cons second rest =>
          simp only [parseAcc, parse]
          split
          · rw [parseAcc_eq rest]
            omega
          · rw [parseAcc_eq (second :: rest)]
            omega
termination_by input.length

theorem parse_positive (input : List String) (nonempty : input ≠ [])
    (valid : ∀ symbol ∈ input, symbol ∈ symbols) : 0 < parse input := by
  cases input with
  | nil => contradiction
  | cons first rest =>
      have positive := value_positive first (valid first (by simp))
      cases rest with
      | nil => simpa only [parse] using positive
      | cons second rest =>
          simp only [parse]
          split <;> omega

/-- Tokens are listed from largest to smallest, including the six subtractive pairs. -/
def tokens : List (Nat × String) :=
  [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

/-- Character-list concatenation permits arbitrary token alphabets. -/
def copies (token : List α) (count : Nat) : List α := (List.replicate count token).flatten

@[simp] theorem copies_zero (token : List α) : copies token 0 = [] := rfl

theorem copies_succ (token : List α) (count : Nat) : copies token (count + 1) = token ++ copies token count := by
  simp [copies, List.replicate_succ]

theorem copies_add (token : List α) (left right : Nat) :
    copies token (left + right) = copies token left ++ copies token right := by
  simp only [copies, List.replicate_add, List.flatten_append]

/-- Division batches exactly the repeated subtractions used by the reference encoder. -/
def encode (table : List (Nat × List α)) (n : Nat) : List α :=
  match table with
  | [] => []
  | (denomination, token) :: rest => copies token (n / denomination) ++ encode rest (n % denomination)

@[simp] theorem encode_zero (table : List (Nat × List α)) : encode table 0 = [] := by
  induction table with
  | nil => rfl
  | cons head rest ih => simp [encode, ih]

theorem encode_skip (denomination n : Nat) (token : List α) (rest : List (Nat × List α))
    (small : n < denomination) :
    encode ((denomination, token) :: rest) n = encode rest n := by
  simp [encode, Nat.div_eq_of_lt small, Nat.mod_eq_of_lt small]

theorem encode_subtract (denomination n : Nat) (token : List α) (rest : List (Nat × List α))
    (positive : 0 < denomination) (enough : denomination ≤ n) :
    encode ((denomination, token) :: rest) n =
      token ++ encode ((denomination, token) :: rest) (n - denomination) := by
  have quotient : n / denomination = (n - denomination) / denomination + 1 := by
    conv_lhs => rw [← Nat.sub_add_cancel enough]
    exact Nat.add_div_right _ positive
  have remainder : n % denomination = (n - denomination) % denomination := by
    conv_lhs => rw [← Nat.sub_add_cancel enough]
    exact Nat.add_mod_right _ _
  simp only [encode, quotient, copies_succ, remainder, List.append_assoc]

def canonical (n : Nat) : List Char := encode (tokens.map (fun pair => (pair.1, pair.2.toList))) n

def savings (input : List String) : Int :=
  (input.map (fun text => (text.length : Int) - (canonical (parse (text.toList.map String.singleton))).length)).sum

end ProjectEulerSolutions.RomanNumerals
