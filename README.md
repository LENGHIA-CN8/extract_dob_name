# Extract Information
This repo use T5 model and regex rule to extract name and dob in customer text <br /> 
|Case| Input | Output |
|---| --- | --- |
|Câu giới thiệu bình thường| 'tôi là huỳnh hiểu minh' | 'huỳnh hiểu minh' |
|Khách hàng không báo tên hoặc câu chứa nhiễu|'ờm không nói cho bạn đâu' | null |
|Bị thiếu số|'ngày tháng năm ờ ờ'| null |
|Trường hợp đọc ngày sinh không đọc từ khóa năm| 'ngày hai tháng một một một chín chín tư' | '02/11/1994' |
|Chứa nhiễu| 'ờ ờ ngày hai mươi tháng ba năm một chín chín tư' | '20/03/1994' |
|Trường hợp nói bị lắp, thừa số, không đúng định dạng| 'ờ ờ ngày hai hai mươi tháng ba năm một chín chín tư' | null |
|Trường hợp đọc ngày sinh không đọc từ khóa ngày, năm| 'hai mươi hai tháng mười một chín chín tư' | '22/10/1994' |
|Trường hợp đọc năm sinh viết tắt| 'ngày hai mươi ba tháng năm năm chín tư' | '23/05/1994' |
|Chứa nhiễu| 'tút tút ờ ờ đào ờ bá lộc nhé' | 'đào bá lộc' |