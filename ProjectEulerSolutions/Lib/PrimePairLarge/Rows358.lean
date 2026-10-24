import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11969_small : List Nat := []
theorem row11969_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11969 row11969_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11969_block_005 : List Nat := []
theorem row11969_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11969 row11969_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11969_block_006 : List Nat := []
def plan11969_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 19), .first, (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row11969_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11969 row11969_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11969_block_006
  · decide
  · rfl

def row11969_block_007 : List Nat := [13397, 13829]
def plan11969_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .second, (.secondFactor 23), .first, .first, .first, (.secondFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .possible, .first, .second, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11969_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11969 row11969_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11969_block_007
  · decide
  · rfl

def row11969_block_008 : List Nat := []
theorem row11969_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11969 row11969_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11969_block_009 : List Nat := []
theorem row11969_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11969 row11969_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11969_block_010 : List Nat := []
theorem row11969_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11969 row11969_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11969_block_011 : List Nat := []
theorem row11969_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11969 row11969_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11969_block_012 : List Nat := []
theorem row11969_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11969 row11969_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11969_block_013 : List Nat := []
theorem row11969_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11969 row11969_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11969_large_left_left : List Nat := row11969_small ++ row11969_block_005
theorem row11969_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11969 row11969_large_left_left :=
  Budget.RowUpperBound.branch row11969_small_covers row11969_block_005_covers

def row11969_large_left_right_right : List Nat := row11969_block_007 ++ row11969_block_008
theorem row11969_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11969 row11969_large_left_right_right :=
  Budget.RowUpperBound.branch row11969_block_007_covers row11969_block_008_covers

def row11969_large_left_right : List Nat := row11969_block_006 ++ row11969_large_left_right_right
theorem row11969_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11969 row11969_large_left_right :=
  Budget.RowUpperBound.branch row11969_block_006_covers row11969_large_left_right_right_covers

def row11969_large_left : List Nat := row11969_large_left_left ++ row11969_large_left_right
theorem row11969_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11969 row11969_large_left :=
  Budget.RowUpperBound.branch row11969_large_left_left_covers row11969_large_left_right_covers

def row11969_large_right_left : List Nat := row11969_block_009 ++ row11969_block_010
theorem row11969_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11969 row11969_large_right_left :=
  Budget.RowUpperBound.branch row11969_block_009_covers row11969_block_010_covers

def row11969_large_right_right_right : List Nat := row11969_block_012 ++ row11969_block_013
theorem row11969_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11969 row11969_large_right_right_right :=
  Budget.RowUpperBound.branch row11969_block_012_covers row11969_block_013_covers

def row11969_large_right_right : List Nat := row11969_block_011 ++ row11969_large_right_right_right
theorem row11969_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11969 row11969_large_right_right :=
  Budget.RowUpperBound.branch row11969_block_011_covers row11969_large_right_right_right_covers

def row11969_large_right : List Nat := row11969_large_right_left ++ row11969_large_right_right
theorem row11969_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11969 row11969_large_right :=
  Budget.RowUpperBound.branch row11969_large_right_left_covers row11969_large_right_right_covers

def row11969_large : List Nat := row11969_large_left ++ row11969_large_right
theorem row11969_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11969 row11969_large :=
  Budget.RowUpperBound.branch row11969_large_left_covers row11969_large_right_covers

def row11969 : List Nat := row11969_large
theorem row11969_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11969 row11969 := row11969_large_covers

def row11971_small : List Nat := []
theorem row11971_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11971 row11971_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11971_block_005 : List Nat := []
theorem row11971_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11971 row11971_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11971_block_006 : List Nat := [12277]
def plan11971_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 37), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 29), .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row11971_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11971 row11971_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11971_block_006
  · decide
  · rfl

def row11971_block_007 : List Nat := [13441, 13681, 13831]
def plan11971_block_007 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .second, (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .second, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .second, .first, .possible, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11971_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11971 row11971_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11971_block_007
  · decide
  · rfl

def row11971_block_008 : List Nat := []
theorem row11971_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11971 row11971_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11971_block_009 : List Nat := []
theorem row11971_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11971 row11971_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11971_block_010 : List Nat := []
theorem row11971_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11971 row11971_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11971_block_011 : List Nat := []
theorem row11971_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11971 row11971_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11971_block_012 : List Nat := []
theorem row11971_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11971 row11971_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11971_block_013 : List Nat := []
theorem row11971_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11971 row11971_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11971_large_left_left : List Nat := row11971_small ++ row11971_block_005
theorem row11971_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11971 row11971_large_left_left :=
  Budget.RowUpperBound.branch row11971_small_covers row11971_block_005_covers

def row11971_large_left_right_right : List Nat := row11971_block_007 ++ row11971_block_008
theorem row11971_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11971 row11971_large_left_right_right :=
  Budget.RowUpperBound.branch row11971_block_007_covers row11971_block_008_covers

def row11971_large_left_right : List Nat := row11971_block_006 ++ row11971_large_left_right_right
theorem row11971_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11971 row11971_large_left_right :=
  Budget.RowUpperBound.branch row11971_block_006_covers row11971_large_left_right_right_covers

def row11971_large_left : List Nat := row11971_large_left_left ++ row11971_large_left_right
theorem row11971_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11971 row11971_large_left :=
  Budget.RowUpperBound.branch row11971_large_left_left_covers row11971_large_left_right_covers

def row11971_large_right_left : List Nat := row11971_block_009 ++ row11971_block_010
theorem row11971_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11971 row11971_large_right_left :=
  Budget.RowUpperBound.branch row11971_block_009_covers row11971_block_010_covers

def row11971_large_right_right_right : List Nat := row11971_block_012 ++ row11971_block_013
theorem row11971_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11971 row11971_large_right_right_right :=
  Budget.RowUpperBound.branch row11971_block_012_covers row11971_block_013_covers

def row11971_large_right_right : List Nat := row11971_block_011 ++ row11971_large_right_right_right
theorem row11971_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11971 row11971_large_right_right :=
  Budget.RowUpperBound.branch row11971_block_011_covers row11971_large_right_right_right_covers

def row11971_large_right : List Nat := row11971_large_right_left ++ row11971_large_right_right
theorem row11971_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11971 row11971_large_right :=
  Budget.RowUpperBound.branch row11971_large_right_left_covers row11971_large_right_right_covers

def row11971_large : List Nat := row11971_large_left ++ row11971_large_right
theorem row11971_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11971 row11971_large :=
  Budget.RowUpperBound.branch row11971_large_left_covers row11971_large_right_covers

def row11971 : List Nat := row11971_large
theorem row11971_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11971 row11971 := row11971_large_covers

def row11981_small : List Nat := []
theorem row11981_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11981 row11981_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11981_block_005 : List Nat := []
theorem row11981_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11981 row11981_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11981_block_006 : List Nat := []
def plan11981_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 29), .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11981_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11981 row11981_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11981_block_006
  · decide
  · rfl

def row11981_block_007 : List Nat := [12611, 13691]
def plan11981_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .second, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 17), .second, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 29), .first, .second, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 29), (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11981_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11981 row11981_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11981_block_007
  · decide
  · rfl

def row11981_block_008 : List Nat := []
theorem row11981_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11981 row11981_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11981_block_009 : List Nat := []
theorem row11981_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11981 row11981_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11981_block_010 : List Nat := []
theorem row11981_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11981 row11981_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11981_block_011 : List Nat := []
theorem row11981_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11981 row11981_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11981_block_012 : List Nat := []
theorem row11981_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11981 row11981_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11981_block_013 : List Nat := []
theorem row11981_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11981 row11981_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11981_large_left_left : List Nat := row11981_small ++ row11981_block_005
theorem row11981_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11981 row11981_large_left_left :=
  Budget.RowUpperBound.branch row11981_small_covers row11981_block_005_covers

def row11981_large_left_right_right : List Nat := row11981_block_007 ++ row11981_block_008
theorem row11981_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11981 row11981_large_left_right_right :=
  Budget.RowUpperBound.branch row11981_block_007_covers row11981_block_008_covers

def row11981_large_left_right : List Nat := row11981_block_006 ++ row11981_large_left_right_right
theorem row11981_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11981 row11981_large_left_right :=
  Budget.RowUpperBound.branch row11981_block_006_covers row11981_large_left_right_right_covers

def row11981_large_left : List Nat := row11981_large_left_left ++ row11981_large_left_right
theorem row11981_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11981 row11981_large_left :=
  Budget.RowUpperBound.branch row11981_large_left_left_covers row11981_large_left_right_covers

def row11981_large_right_left : List Nat := row11981_block_009 ++ row11981_block_010
theorem row11981_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11981 row11981_large_right_left :=
  Budget.RowUpperBound.branch row11981_block_009_covers row11981_block_010_covers

def row11981_large_right_right_right : List Nat := row11981_block_012 ++ row11981_block_013
theorem row11981_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11981 row11981_large_right_right_right :=
  Budget.RowUpperBound.branch row11981_block_012_covers row11981_block_013_covers

def row11981_large_right_right : List Nat := row11981_block_011 ++ row11981_large_right_right_right
theorem row11981_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11981 row11981_large_right_right :=
  Budget.RowUpperBound.branch row11981_block_011_covers row11981_large_right_right_right_covers

def row11981_large_right : List Nat := row11981_large_right_left ++ row11981_large_right_right
theorem row11981_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11981 row11981_large_right :=
  Budget.RowUpperBound.branch row11981_large_right_left_covers row11981_large_right_right_covers

def row11981_large : List Nat := row11981_large_left ++ row11981_large_right
theorem row11981_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11981 row11981_large :=
  Budget.RowUpperBound.branch row11981_large_left_covers row11981_large_right_covers

def row11981 : List Nat := row11981_large
theorem row11981_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11981 row11981 := row11981_large_covers

def row11987_small : List Nat := []
theorem row11987_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11987 row11987_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11987_block_005 : List Nat := []
theorem row11987_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11987 row11987_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11987_block_006 : List Nat := []
def plan11987_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row11987_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11987 row11987_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11987_block_006
  · decide
  · rfl

def row11987_block_007 : List Nat := [13127]
def plan11987_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), (.secondFactor 17), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 37), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 37), (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11987_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11987 row11987_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11987_block_007
  · decide
  · rfl

def row11987_block_008 : List Nat := []
theorem row11987_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11987 row11987_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11987_block_009 : List Nat := []
theorem row11987_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11987 row11987_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11987_block_010 : List Nat := []
theorem row11987_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11987 row11987_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11987_block_011 : List Nat := []
theorem row11987_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11987 row11987_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11987_block_012 : List Nat := []
theorem row11987_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11987 row11987_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11987_block_013 : List Nat := []
theorem row11987_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11987 row11987_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11987_large_left_left : List Nat := row11987_small ++ row11987_block_005
theorem row11987_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11987 row11987_large_left_left :=
  Budget.RowUpperBound.branch row11987_small_covers row11987_block_005_covers

def row11987_large_left_right_right : List Nat := row11987_block_007 ++ row11987_block_008
theorem row11987_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11987 row11987_large_left_right_right :=
  Budget.RowUpperBound.branch row11987_block_007_covers row11987_block_008_covers

def row11987_large_left_right : List Nat := row11987_block_006 ++ row11987_large_left_right_right
theorem row11987_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11987 row11987_large_left_right :=
  Budget.RowUpperBound.branch row11987_block_006_covers row11987_large_left_right_right_covers

def row11987_large_left : List Nat := row11987_large_left_left ++ row11987_large_left_right
theorem row11987_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11987 row11987_large_left :=
  Budget.RowUpperBound.branch row11987_large_left_left_covers row11987_large_left_right_covers

def row11987_large_right_left : List Nat := row11987_block_009 ++ row11987_block_010
theorem row11987_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11987 row11987_large_right_left :=
  Budget.RowUpperBound.branch row11987_block_009_covers row11987_block_010_covers

def row11987_large_right_right_right : List Nat := row11987_block_012 ++ row11987_block_013
theorem row11987_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11987 row11987_large_right_right_right :=
  Budget.RowUpperBound.branch row11987_block_012_covers row11987_block_013_covers

def row11987_large_right_right : List Nat := row11987_block_011 ++ row11987_large_right_right_right
theorem row11987_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11987 row11987_large_right_right :=
  Budget.RowUpperBound.branch row11987_block_011_covers row11987_large_right_right_right_covers

def row11987_large_right : List Nat := row11987_large_right_left ++ row11987_large_right_right
theorem row11987_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11987 row11987_large_right :=
  Budget.RowUpperBound.branch row11987_large_right_left_covers row11987_large_right_right_covers

def row11987_large : List Nat := row11987_large_left ++ row11987_large_right
theorem row11987_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11987 row11987_large :=
  Budget.RowUpperBound.branch row11987_large_left_covers row11987_large_right_covers

def row11987 : List Nat := row11987_large
theorem row11987_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11987 row11987 := row11987_large_covers

end ProjectEulerSolutions.PrimePairLarge
