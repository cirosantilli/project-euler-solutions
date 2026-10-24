import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12203_small : List Nat := []
theorem row12203_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12203 row12203_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12203_block_005 : List Nat := []
theorem row12203_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12203 row12203_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12203_block_006 : List Nat := []
def plan12203_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row12203_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12203 row12203_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12203_block_006
  · decide
  · rfl

def row12203_block_007 : List Nat := [13421]
def plan12203_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .second, .second, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .second, .first, .first, .first, (.secondFactor 37), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12203_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12203 row12203_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12203_block_007
  · decide
  · rfl

def row12203_block_008 : List Nat := []
theorem row12203_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12203 row12203_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12203_block_009 : List Nat := []
theorem row12203_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12203 row12203_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12203_block_010 : List Nat := []
theorem row12203_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12203 row12203_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12203_block_011 : List Nat := []
theorem row12203_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12203 row12203_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12203_block_012 : List Nat := []
theorem row12203_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12203 row12203_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12203_block_013 : List Nat := []
theorem row12203_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12203 row12203_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12203_large_left_left : List Nat := row12203_small ++ row12203_block_005
theorem row12203_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12203 row12203_large_left_left :=
  Budget.RowUpperBound.branch row12203_small_covers row12203_block_005_covers

def row12203_large_left_right_right : List Nat := row12203_block_007 ++ row12203_block_008
theorem row12203_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12203 row12203_large_left_right_right :=
  Budget.RowUpperBound.branch row12203_block_007_covers row12203_block_008_covers

def row12203_large_left_right : List Nat := row12203_block_006 ++ row12203_large_left_right_right
theorem row12203_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12203 row12203_large_left_right :=
  Budget.RowUpperBound.branch row12203_block_006_covers row12203_large_left_right_right_covers

def row12203_large_left : List Nat := row12203_large_left_left ++ row12203_large_left_right
theorem row12203_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12203 row12203_large_left :=
  Budget.RowUpperBound.branch row12203_large_left_left_covers row12203_large_left_right_covers

def row12203_large_right_left : List Nat := row12203_block_009 ++ row12203_block_010
theorem row12203_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12203 row12203_large_right_left :=
  Budget.RowUpperBound.branch row12203_block_009_covers row12203_block_010_covers

def row12203_large_right_right_right : List Nat := row12203_block_012 ++ row12203_block_013
theorem row12203_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12203 row12203_large_right_right_right :=
  Budget.RowUpperBound.branch row12203_block_012_covers row12203_block_013_covers

def row12203_large_right_right : List Nat := row12203_block_011 ++ row12203_large_right_right_right
theorem row12203_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12203 row12203_large_right_right :=
  Budget.RowUpperBound.branch row12203_block_011_covers row12203_large_right_right_right_covers

def row12203_large_right : List Nat := row12203_large_right_left ++ row12203_large_right_right
theorem row12203_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12203 row12203_large_right :=
  Budget.RowUpperBound.branch row12203_large_right_left_covers row12203_large_right_right_covers

def row12203_large : List Nat := row12203_large_left ++ row12203_large_right
theorem row12203_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12203 row12203_large :=
  Budget.RowUpperBound.branch row12203_large_left_covers row12203_large_right_covers

def row12203 : List Nat := row12203_large
theorem row12203_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12203 row12203 := row12203_large_covers

def row12211_small : List Nat := []
theorem row12211_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12211 row12211_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12211_block_005 : List Nat := []
theorem row12211_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12211 row12211_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12211_block_006 : List Nat := []
def plan12211_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row12211_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12211 row12211_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12211_block_006
  · decide
  · rfl

def row12211_block_007 : List Nat := [13633, 13807]
def plan12211_block_007 : List Budget.Choice := [.first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 19), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 17), (.firstFactor 37), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 37), .possible, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12211_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12211 row12211_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12211_block_007
  · decide
  · rfl

def row12211_block_008 : List Nat := []
theorem row12211_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12211 row12211_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12211_block_009 : List Nat := []
theorem row12211_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12211 row12211_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12211_block_010 : List Nat := []
theorem row12211_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12211 row12211_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12211_block_011 : List Nat := []
theorem row12211_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12211 row12211_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12211_block_012 : List Nat := []
theorem row12211_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12211 row12211_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12211_block_013 : List Nat := []
theorem row12211_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12211 row12211_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12211_large_left_left : List Nat := row12211_small ++ row12211_block_005
theorem row12211_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12211 row12211_large_left_left :=
  Budget.RowUpperBound.branch row12211_small_covers row12211_block_005_covers

def row12211_large_left_right_right : List Nat := row12211_block_007 ++ row12211_block_008
theorem row12211_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12211 row12211_large_left_right_right :=
  Budget.RowUpperBound.branch row12211_block_007_covers row12211_block_008_covers

def row12211_large_left_right : List Nat := row12211_block_006 ++ row12211_large_left_right_right
theorem row12211_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12211 row12211_large_left_right :=
  Budget.RowUpperBound.branch row12211_block_006_covers row12211_large_left_right_right_covers

def row12211_large_left : List Nat := row12211_large_left_left ++ row12211_large_left_right
theorem row12211_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12211 row12211_large_left :=
  Budget.RowUpperBound.branch row12211_large_left_left_covers row12211_large_left_right_covers

def row12211_large_right_left : List Nat := row12211_block_009 ++ row12211_block_010
theorem row12211_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12211 row12211_large_right_left :=
  Budget.RowUpperBound.branch row12211_block_009_covers row12211_block_010_covers

def row12211_large_right_right_right : List Nat := row12211_block_012 ++ row12211_block_013
theorem row12211_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12211 row12211_large_right_right_right :=
  Budget.RowUpperBound.branch row12211_block_012_covers row12211_block_013_covers

def row12211_large_right_right : List Nat := row12211_block_011 ++ row12211_large_right_right_right
theorem row12211_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12211 row12211_large_right_right :=
  Budget.RowUpperBound.branch row12211_block_011_covers row12211_large_right_right_right_covers

def row12211_large_right : List Nat := row12211_large_right_left ++ row12211_large_right_right
theorem row12211_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12211 row12211_large_right :=
  Budget.RowUpperBound.branch row12211_large_right_left_covers row12211_large_right_right_covers

def row12211_large : List Nat := row12211_large_left ++ row12211_large_right
theorem row12211_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12211 row12211_large :=
  Budget.RowUpperBound.branch row12211_large_left_covers row12211_large_right_covers

def row12211 : List Nat := row12211_large
theorem row12211_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12211 row12211 := row12211_large_covers

def row12227_small : List Nat := []
theorem row12227_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12227 row12227_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12227_block_005 : List Nat := []
theorem row12227_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12227 row12227_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12227_block_006 : List Nat := []
def plan12227_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row12227_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12227 row12227_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12227_block_006
  · decide
  · rfl

def row12227_block_007 : List Nat := [12479, 12671, 13001]
def plan12227_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12227_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12227 row12227_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12227_block_007
  · decide
  · rfl

def row12227_block_008 : List Nat := []
theorem row12227_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12227 row12227_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12227_block_009 : List Nat := []
theorem row12227_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12227 row12227_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12227_block_010 : List Nat := []
theorem row12227_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12227 row12227_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12227_block_011 : List Nat := []
theorem row12227_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12227 row12227_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12227_block_012 : List Nat := []
theorem row12227_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12227 row12227_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12227_block_013 : List Nat := []
theorem row12227_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12227 row12227_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12227_large_left_left : List Nat := row12227_small ++ row12227_block_005
theorem row12227_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12227 row12227_large_left_left :=
  Budget.RowUpperBound.branch row12227_small_covers row12227_block_005_covers

def row12227_large_left_right_right : List Nat := row12227_block_007 ++ row12227_block_008
theorem row12227_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12227 row12227_large_left_right_right :=
  Budget.RowUpperBound.branch row12227_block_007_covers row12227_block_008_covers

def row12227_large_left_right : List Nat := row12227_block_006 ++ row12227_large_left_right_right
theorem row12227_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12227 row12227_large_left_right :=
  Budget.RowUpperBound.branch row12227_block_006_covers row12227_large_left_right_right_covers

def row12227_large_left : List Nat := row12227_large_left_left ++ row12227_large_left_right
theorem row12227_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12227 row12227_large_left :=
  Budget.RowUpperBound.branch row12227_large_left_left_covers row12227_large_left_right_covers

def row12227_large_right_left : List Nat := row12227_block_009 ++ row12227_block_010
theorem row12227_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12227 row12227_large_right_left :=
  Budget.RowUpperBound.branch row12227_block_009_covers row12227_block_010_covers

def row12227_large_right_right_right : List Nat := row12227_block_012 ++ row12227_block_013
theorem row12227_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12227 row12227_large_right_right_right :=
  Budget.RowUpperBound.branch row12227_block_012_covers row12227_block_013_covers

def row12227_large_right_right : List Nat := row12227_block_011 ++ row12227_large_right_right_right
theorem row12227_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12227 row12227_large_right_right :=
  Budget.RowUpperBound.branch row12227_block_011_covers row12227_large_right_right_right_covers

def row12227_large_right : List Nat := row12227_large_right_left ++ row12227_large_right_right
theorem row12227_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12227 row12227_large_right :=
  Budget.RowUpperBound.branch row12227_large_right_left_covers row12227_large_right_right_covers

def row12227_large : List Nat := row12227_large_left ++ row12227_large_right
theorem row12227_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12227 row12227_large :=
  Budget.RowUpperBound.branch row12227_large_left_covers row12227_large_right_covers

def row12227 : List Nat := row12227_large
theorem row12227_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12227 row12227 := row12227_large_covers

def row12239_small : List Nat := []
theorem row12239_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12239 row12239_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12239_block_005 : List Nat := []
theorem row12239_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12239 row12239_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12239_block_006 : List Nat := []
def plan12239_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row12239_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12239 row12239_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12239_block_006
  · decide
  · rfl

def row12239_block_007 : List Nat := [13463]
def plan12239_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 19), (.firstFactor 13), .possible, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12239_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12239 row12239_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12239_block_007
  · decide
  · rfl

def row12239_block_008 : List Nat := []
theorem row12239_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12239 row12239_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12239_block_009 : List Nat := []
theorem row12239_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12239 row12239_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12239_block_010 : List Nat := []
theorem row12239_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12239 row12239_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12239_block_011 : List Nat := []
theorem row12239_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12239 row12239_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12239_block_012 : List Nat := []
theorem row12239_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12239 row12239_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12239_block_013 : List Nat := []
theorem row12239_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12239 row12239_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12239_large_left_left : List Nat := row12239_small ++ row12239_block_005
theorem row12239_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12239 row12239_large_left_left :=
  Budget.RowUpperBound.branch row12239_small_covers row12239_block_005_covers

def row12239_large_left_right_right : List Nat := row12239_block_007 ++ row12239_block_008
theorem row12239_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12239 row12239_large_left_right_right :=
  Budget.RowUpperBound.branch row12239_block_007_covers row12239_block_008_covers

def row12239_large_left_right : List Nat := row12239_block_006 ++ row12239_large_left_right_right
theorem row12239_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12239 row12239_large_left_right :=
  Budget.RowUpperBound.branch row12239_block_006_covers row12239_large_left_right_right_covers

def row12239_large_left : List Nat := row12239_large_left_left ++ row12239_large_left_right
theorem row12239_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12239 row12239_large_left :=
  Budget.RowUpperBound.branch row12239_large_left_left_covers row12239_large_left_right_covers

def row12239_large_right_left : List Nat := row12239_block_009 ++ row12239_block_010
theorem row12239_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12239 row12239_large_right_left :=
  Budget.RowUpperBound.branch row12239_block_009_covers row12239_block_010_covers

def row12239_large_right_right_right : List Nat := row12239_block_012 ++ row12239_block_013
theorem row12239_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12239 row12239_large_right_right_right :=
  Budget.RowUpperBound.branch row12239_block_012_covers row12239_block_013_covers

def row12239_large_right_right : List Nat := row12239_block_011 ++ row12239_large_right_right_right
theorem row12239_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12239 row12239_large_right_right :=
  Budget.RowUpperBound.branch row12239_block_011_covers row12239_large_right_right_right_covers

def row12239_large_right : List Nat := row12239_large_right_left ++ row12239_large_right_right
theorem row12239_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12239 row12239_large_right :=
  Budget.RowUpperBound.branch row12239_large_right_left_covers row12239_large_right_right_covers

def row12239_large : List Nat := row12239_large_left ++ row12239_large_right
theorem row12239_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12239 row12239_large :=
  Budget.RowUpperBound.branch row12239_large_left_covers row12239_large_right_covers

def row12239 : List Nat := row12239_large
theorem row12239_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12239 row12239 := row12239_large_covers

end ProjectEulerSolutions.PrimePairLarge
