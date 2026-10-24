import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8719_small : List Nat := []
theorem row8719_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8719 row8719_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8719_block_005 : List Nat := [9649, 9901]
def plan8719_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 37), (.firstFactor 23), .second, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 13), .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 29), (.firstFactor 37), .first, .first, .first, (.firstFactor 17), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8719_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8719 row8719_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8719_block_005
  · decide
  · rfl

def row8719_block_006 : List Nat := [10987, 11119, 11701, 11839, 11923, 12409]
def plan8719_block_006 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 23), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), .first, .first, .possible, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row8719_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8719 row8719_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8719_block_006
  · decide
  · rfl

def row8719_block_007 : List Nat := [12703, 12889, 13723, 14341]
def plan8719_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 11), .second, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .possible, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 37), .second, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .second, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8719_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8719 row8719_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8719_block_007
  · decide
  · rfl

def row8719_block_008 : List Nat := [14737, 14779, 15271, 15313, 15391, 15559, 15739, 15817, 15991, 16189, 16231]
def plan8719_block_008 : List Budget.Choice := [(.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .possible, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .possible, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .second, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8719_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8719 row8719_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8719_block_008
  · decide
  · rfl

def row8719_block_009 : List Nat := [16699, 16993]
def plan8719_block_009 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .possible, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 29), .possible, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8719_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8719 row8719_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8719_block_009
  · decide
  · rfl

def row8719_block_010 : List Nat := []
theorem row8719_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8719 row8719_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8719_block_011 : List Nat := []
theorem row8719_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8719 row8719_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8719_block_012 : List Nat := []
theorem row8719_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8719 row8719_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8719_block_013 : List Nat := []
theorem row8719_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8719 row8719_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8719_large_left_left : List Nat := row8719_small ++ row8719_block_005
theorem row8719_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8719 row8719_large_left_left :=
  Budget.RowUpperBound.branch row8719_small_covers row8719_block_005_covers

def row8719_large_left_right_right : List Nat := row8719_block_007 ++ row8719_block_008
theorem row8719_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8719 row8719_large_left_right_right :=
  Budget.RowUpperBound.branch row8719_block_007_covers row8719_block_008_covers

def row8719_large_left_right : List Nat := row8719_block_006 ++ row8719_large_left_right_right
theorem row8719_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8719 row8719_large_left_right :=
  Budget.RowUpperBound.branch row8719_block_006_covers row8719_large_left_right_right_covers

def row8719_large_left : List Nat := row8719_large_left_left ++ row8719_large_left_right
theorem row8719_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8719 row8719_large_left :=
  Budget.RowUpperBound.branch row8719_large_left_left_covers row8719_large_left_right_covers

def row8719_large_right_left : List Nat := row8719_block_009 ++ row8719_block_010
theorem row8719_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8719 row8719_large_right_left :=
  Budget.RowUpperBound.branch row8719_block_009_covers row8719_block_010_covers

def row8719_large_right_right_right : List Nat := row8719_block_012 ++ row8719_block_013
theorem row8719_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8719 row8719_large_right_right_right :=
  Budget.RowUpperBound.branch row8719_block_012_covers row8719_block_013_covers

def row8719_large_right_right : List Nat := row8719_block_011 ++ row8719_large_right_right_right
theorem row8719_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8719 row8719_large_right_right :=
  Budget.RowUpperBound.branch row8719_block_011_covers row8719_large_right_right_right_covers

def row8719_large_right : List Nat := row8719_large_right_left ++ row8719_large_right_right
theorem row8719_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8719 row8719_large_right :=
  Budget.RowUpperBound.branch row8719_large_right_left_covers row8719_large_right_right_covers

def row8719_large : List Nat := row8719_large_left ++ row8719_large_right
theorem row8719_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8719 row8719_large :=
  Budget.RowUpperBound.branch row8719_large_left_covers row8719_large_right_covers

def row8719 : List Nat := row8719_large
theorem row8719_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8719 row8719 := row8719_large_covers

def row8731_small : List Nat := []
theorem row8731_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8731 row8731_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8731_block_005 : List Nat := [9007, 9151, 9391, 9433, 9949, 10093, 10099, 10177, 10333, 10369]
def plan8731_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .second, (.secondFactor 7), .second, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .possible, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 37), (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .second, .first, .first, (.firstFactor 13), .first, .first, .possible, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .possible]
set_option maxRecDepth 2048 in
theorem row8731_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8731 row8731_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8731_block_005
  · decide
  · rfl

def row8731_block_006 : List Nat := [10723, 11299, 12157]
def plan8731_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .second, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8731_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8731 row8731_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8731_block_006
  · decide
  · rfl

def row8731_block_007 : List Nat := [12973, 13711]
def plan8731_block_007 : List Budget.Choice := [(.firstFactor 37), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 13), (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8731_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8731 row8731_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8731_block_007
  · decide
  · rfl

def row8731_block_008 : List Nat := [14503, 14887, 14929, 15013]
def plan8731_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 19), .possible, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 31), (.secondFactor 23), .first, .first, (.firstFactor 31), (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8731_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8731 row8731_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8731_block_008
  · decide
  · rfl

def row8731_block_009 : List Nat := []
def plan8731_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8731_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8731 row8731_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8731_block_009
  · decide
  · rfl

def row8731_block_010 : List Nat := []
theorem row8731_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8731 row8731_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8731_block_011 : List Nat := []
theorem row8731_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8731 row8731_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8731_block_012 : List Nat := []
theorem row8731_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8731 row8731_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8731_block_013 : List Nat := []
theorem row8731_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8731 row8731_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8731_large_left_left : List Nat := row8731_small ++ row8731_block_005
theorem row8731_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8731 row8731_large_left_left :=
  Budget.RowUpperBound.branch row8731_small_covers row8731_block_005_covers

def row8731_large_left_right_right : List Nat := row8731_block_007 ++ row8731_block_008
theorem row8731_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8731 row8731_large_left_right_right :=
  Budget.RowUpperBound.branch row8731_block_007_covers row8731_block_008_covers

def row8731_large_left_right : List Nat := row8731_block_006 ++ row8731_large_left_right_right
theorem row8731_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8731 row8731_large_left_right :=
  Budget.RowUpperBound.branch row8731_block_006_covers row8731_large_left_right_right_covers

def row8731_large_left : List Nat := row8731_large_left_left ++ row8731_large_left_right
theorem row8731_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8731 row8731_large_left :=
  Budget.RowUpperBound.branch row8731_large_left_left_covers row8731_large_left_right_covers

def row8731_large_right_left : List Nat := row8731_block_009 ++ row8731_block_010
theorem row8731_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8731 row8731_large_right_left :=
  Budget.RowUpperBound.branch row8731_block_009_covers row8731_block_010_covers

def row8731_large_right_right_right : List Nat := row8731_block_012 ++ row8731_block_013
theorem row8731_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8731 row8731_large_right_right_right :=
  Budget.RowUpperBound.branch row8731_block_012_covers row8731_block_013_covers

def row8731_large_right_right : List Nat := row8731_block_011 ++ row8731_large_right_right_right
theorem row8731_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8731 row8731_large_right_right :=
  Budget.RowUpperBound.branch row8731_block_011_covers row8731_large_right_right_right_covers

def row8731_large_right : List Nat := row8731_large_right_left ++ row8731_large_right_right
theorem row8731_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8731 row8731_large_right :=
  Budget.RowUpperBound.branch row8731_large_right_left_covers row8731_large_right_right_covers

def row8731_large : List Nat := row8731_large_left ++ row8731_large_right
theorem row8731_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8731 row8731_large :=
  Budget.RowUpperBound.branch row8731_large_left_covers row8731_large_right_covers

def row8731 : List Nat := row8731_large
theorem row8731_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8731 row8731 := row8731_large_covers

def row8737_small : List Nat := []
theorem row8737_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8737 row8737_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8737_block_005 : List Nat := [9151, 9739, 9883, 10069]
def plan8737_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 31), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .possible, (.firstFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 23), .first, .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .possible, .first, .first, .first, .second, .second, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .possible, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .second, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8737_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8737 row8737_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8737_block_005
  · decide
  · rfl

def row8737_block_006 : List Nat := [11299, 11491, 11953]
def plan8737_block_006 : List Budget.Choice := [.first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), (.secondFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .second, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .possible, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .possible, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .possible, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 19), .second, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8737_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8737 row8737_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8737_block_006
  · decide
  · rfl

def row8737_block_007 : List Nat := [12517]
def plan8737_block_007 : List Budget.Choice := [.second, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 13), (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 29), (.firstFactor 37), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row8737_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8737 row8737_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8737_block_007
  · decide
  · rfl

def row8737_block_008 : List Nat := [14533, 14683, 14767, 15061, 15091, 15121, 15607, 15649, 16087]
def plan8737_block_008 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), .first, .first, .possible, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8737_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8737 row8737_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8737_block_008
  · decide
  · rfl

def row8737_block_009 : List Nat := [17293]
def plan8737_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 31), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8737_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8737 row8737_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8737_block_009
  · decide
  · rfl

def row8737_block_010 : List Nat := []
theorem row8737_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8737 row8737_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8737_block_011 : List Nat := []
theorem row8737_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8737 row8737_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8737_block_012 : List Nat := []
theorem row8737_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8737 row8737_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8737_block_013 : List Nat := []
theorem row8737_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8737 row8737_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8737_large_left_left : List Nat := row8737_small ++ row8737_block_005
theorem row8737_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8737 row8737_large_left_left :=
  Budget.RowUpperBound.branch row8737_small_covers row8737_block_005_covers

def row8737_large_left_right_right : List Nat := row8737_block_007 ++ row8737_block_008
theorem row8737_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8737 row8737_large_left_right_right :=
  Budget.RowUpperBound.branch row8737_block_007_covers row8737_block_008_covers

def row8737_large_left_right : List Nat := row8737_block_006 ++ row8737_large_left_right_right
theorem row8737_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8737 row8737_large_left_right :=
  Budget.RowUpperBound.branch row8737_block_006_covers row8737_large_left_right_right_covers

def row8737_large_left : List Nat := row8737_large_left_left ++ row8737_large_left_right
theorem row8737_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8737 row8737_large_left :=
  Budget.RowUpperBound.branch row8737_large_left_left_covers row8737_large_left_right_covers

def row8737_large_right_left : List Nat := row8737_block_009 ++ row8737_block_010
theorem row8737_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8737 row8737_large_right_left :=
  Budget.RowUpperBound.branch row8737_block_009_covers row8737_block_010_covers

def row8737_large_right_right_right : List Nat := row8737_block_012 ++ row8737_block_013
theorem row8737_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8737 row8737_large_right_right_right :=
  Budget.RowUpperBound.branch row8737_block_012_covers row8737_block_013_covers

def row8737_large_right_right : List Nat := row8737_block_011 ++ row8737_large_right_right_right
theorem row8737_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8737 row8737_large_right_right :=
  Budget.RowUpperBound.branch row8737_block_011_covers row8737_large_right_right_right_covers

def row8737_large_right : List Nat := row8737_large_right_left ++ row8737_large_right_right
theorem row8737_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8737 row8737_large_right :=
  Budget.RowUpperBound.branch row8737_large_right_left_covers row8737_large_right_right_covers

def row8737_large : List Nat := row8737_large_left ++ row8737_large_right
theorem row8737_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8737 row8737_large :=
  Budget.RowUpperBound.branch row8737_large_left_covers row8737_large_right_covers

def row8737 : List Nat := row8737_large
theorem row8737_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8737 row8737 := row8737_large_covers

def row8741_small : List Nat := []
theorem row8741_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8741 row8741_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8741_block_005 : List Nat := [9239, 9677, 10151]
def plan8741_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .possible, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8741_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8741 row8741_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8741_block_005
  · decide
  · rfl

def row8741_block_006 : List Nat := [11471, 11633, 11789]
def plan8741_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .second, .possible, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .possible, .second, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row8741_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8741 row8741_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8741_block_006
  · decide
  · rfl

def row8741_block_007 : List Nat := [13109, 13313]
def plan8741_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .second, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 17), .first, (.secondFactor 31), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8741_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8741 row8741_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8741_block_007
  · decide
  · rfl

def row8741_block_008 : List Nat := [14771, 14783, 15233, 15461, 15731, 16301]
def plan8741_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .possible, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, .second, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8741_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8741 row8741_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8741_block_008
  · decide
  · rfl

def row8741_block_009 : List Nat := [17117]
def plan8741_block_009 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 37), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 31), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 29), .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8741_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8741 row8741_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8741_block_009
  · decide
  · rfl

def row8741_block_010 : List Nat := []
theorem row8741_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8741 row8741_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8741_block_011 : List Nat := []
theorem row8741_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8741 row8741_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8741_block_012 : List Nat := []
theorem row8741_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8741 row8741_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8741_block_013 : List Nat := []
theorem row8741_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8741 row8741_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8741_large_left_left : List Nat := row8741_small ++ row8741_block_005
theorem row8741_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8741 row8741_large_left_left :=
  Budget.RowUpperBound.branch row8741_small_covers row8741_block_005_covers

def row8741_large_left_right_right : List Nat := row8741_block_007 ++ row8741_block_008
theorem row8741_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8741 row8741_large_left_right_right :=
  Budget.RowUpperBound.branch row8741_block_007_covers row8741_block_008_covers

def row8741_large_left_right : List Nat := row8741_block_006 ++ row8741_large_left_right_right
theorem row8741_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8741 row8741_large_left_right :=
  Budget.RowUpperBound.branch row8741_block_006_covers row8741_large_left_right_right_covers

def row8741_large_left : List Nat := row8741_large_left_left ++ row8741_large_left_right
theorem row8741_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8741 row8741_large_left :=
  Budget.RowUpperBound.branch row8741_large_left_left_covers row8741_large_left_right_covers

def row8741_large_right_left : List Nat := row8741_block_009 ++ row8741_block_010
theorem row8741_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8741 row8741_large_right_left :=
  Budget.RowUpperBound.branch row8741_block_009_covers row8741_block_010_covers

def row8741_large_right_right_right : List Nat := row8741_block_012 ++ row8741_block_013
theorem row8741_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8741 row8741_large_right_right_right :=
  Budget.RowUpperBound.branch row8741_block_012_covers row8741_block_013_covers

def row8741_large_right_right : List Nat := row8741_block_011 ++ row8741_large_right_right_right
theorem row8741_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8741 row8741_large_right_right :=
  Budget.RowUpperBound.branch row8741_block_011_covers row8741_large_right_right_right_covers

def row8741_large_right : List Nat := row8741_large_right_left ++ row8741_large_right_right
theorem row8741_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8741 row8741_large_right :=
  Budget.RowUpperBound.branch row8741_large_right_left_covers row8741_large_right_right_covers

def row8741_large : List Nat := row8741_large_left ++ row8741_large_right
theorem row8741_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8741 row8741_large :=
  Budget.RowUpperBound.branch row8741_large_left_covers row8741_large_right_covers

def row8741 : List Nat := row8741_large
theorem row8741_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8741 row8741 := row8741_large_covers

end ProjectEulerSolutions.PrimePairLarge
