import 'dart:convert';
import 'dart:io';
import 'package:llamadart/src/core/template/chat_template_engine.dart';
import 'package:llamadart/llamadart.dart' show LlamaChatMessage, LlamaChatRole;

void main(List<String> a) {
  final m = jsonDecode(File(a[0]).readAsStringSync()) as Map<String, dynamic>;
  final prompt = m['prompt'] as String;
  for (final g in ['yi15-6b', 'llama2-7b', 'deepseek7b']) {
    final md = (m[g] as Map).map((k, v) => MapEntry(k as String, v as String));
    final r = ChatTemplateEngine.render(
      templateSource: md['tokenizer.chat_template'],
      messages: [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt)],
      metadata: md,
      addAssistant: true,
      enableThinking: false,
    );
    final p = r.prompt;
    print('$g | format ${r.format} | length ${p.length} | head ${jsonEncode(p.substring(0, 40))} | tail ${jsonEncode(p.substring(p.length - 40))}');
    File('${a[1]}/rendered-$g.txt').writeAsStringSync(p);
  }
}
